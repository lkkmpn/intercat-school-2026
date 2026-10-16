import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from ase.atom import Atom
from ase.atoms import Atoms
from ase.build.surface import fcc211
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints.fix_atoms import FixAtoms
from ase.geometry import get_distances
from ase.io import write
from ase.mep.neb import NEB
from ase.optimize.bfgs import BFGS
from ase.optimize.mdmin import MDMin
from ase.optimize.optimize import Optimizer
from dscribe.descriptors.soap import SOAP
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import squareform
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from ics26.ml_methods._internal.mlip import get_uma_calculator
from ics26.ml_methods.gpr import SGPR, SOAPGPRCalculator

ROOT_PATH = Path(__file__).parent.parent.parent.parent.parent / 'files/ml-data'


def generate_gold_stepped() -> Atoms:
    surface = fcc211('Au', (6, 6, 4), vacuum=10.0)

    zs = np.unique(surface.positions[:, 2])
    surface.set_constraint(FixAtoms(mask=surface.positions[:, 2] < zs[-4]))

    return surface


def generate_gold_stepped_small() -> Atoms:
    surface = fcc211('Au', (6, 4, 3), vacuum=10.0)
    surface.set_constraint(FixAtoms(range(len(surface))))

    return surface


def generate_gold_kinked() -> Atoms:
    surface = generate_gold_stepped_small()

    mask = np.ones(len(surface), dtype=bool)
    mask[[1, 2]] = False
    surface = surface[mask]

    return surface


def generate_hydrogen_on_gold_stepped():
    gold_stepped = generate_gold_stepped()

    calc = get_uma_calculator()

    def _relax(between: list[int]) -> Atoms:
        initial_h_pos = np.mean(gold_stepped.positions[between], axis=0) + [0.0, 0.0, 1.0]
        atoms = gold_stepped + Atoms('H', positions=[initial_h_pos])
        atoms.calc = calc
        BFGS(atoms).run(fmax=0.01)
        return atoms

    start = _relax([2, 3])

    write(ROOT_PATH / 'gold_stepped_hydrogen.traj', start)

    # generate unique along a vector
    a_atoms = [_relax([32, 33]), _relax([21])]

    # transpose along a vector
    x_offset = start.cell[0, 0] / 2
    for i in range(3):
        a = [start, *a_atoms][i].copy()
        a.positions[:, 0] += x_offset
        a_atoms.append(reindex(wrap(a), start))

    a_atoms = [start.copy(), *a_atoms]
    for i, atoms in enumerate(a_atoms):
        atoms.info['label'] = f'A{i + 1}'

    write(ROOT_PATH / 'gold_stepped_hydrogen_x.traj', a_atoms)

    # generate along b vector
    y_offset = start.cell[1, 1] / 6
    b_atoms: list[Atoms] = []
    for i in range(6):
        b = start.copy()
        b.positions[:, 1] += i * y_offset
        b = reindex(wrap(b), start)
        b.info['label'] = f'B{i + 1}'
        b_atoms.append(b)

    write(ROOT_PATH / 'gold_stepped_hydrogen_y.traj', b_atoms)

    def _neb(initial: Atoms, final: Atoms):
        images = [initial] + [initial.copy() for _ in range(19)] + [final]
        for image in images[1:-1]:
            if 'label' in image.info:
                del image.info['label']
        neb = NEB(images, allow_shared_calculator=True, method='improvedtangent')
        neb.interpolate(mic=True)
        for image in images:
            image.calc = calc
        MDMin(neb).run(fmax=0.1)
        return images

    # generate neb trajectories along a vector
    neb_traj_a = (
        _neb(a_atoms[0], a_atoms[1])[:-1] + _neb(a_atoms[1], a_atoms[2])[:-1] + _neb(a_atoms[2], a_atoms[3])[:-1]
    )

    for i in range(len(neb_traj_a)):
        image = neb_traj_a[i].copy()
        image.positions[:, 0] += x_offset
        image = reindex(wrap(image), start)
        if 'label' in image.info:
            label_i = int(image.info['label'].lstrip('A'))
            image.info['label'] = f'A{label_i + 3}'
            print(label_i, image.info['label'])
        neb_traj_a.append(image)

    write(ROOT_PATH / 'gold_stepped_hydrogen_x_diffusion.traj', neb_traj_a)


def generate_kinked_training_data():
    calc = get_uma_calculator()

    surface = generate_gold_kinked()
    surface.calc = calc
    e_surf = surface.get_potential_energy()

    h = Atoms('H')
    h.calc = calc
    e_h = h.get_potential_energy()  # precomputed DFT reference

    atoms_random = generate_kinked_training_data_random_single_point(calc)
    atoms_relaxed, atoms_trajectory = generate_kinked_training_data_relaxed_grid(calc)

    training_data = atoms_random + atoms_relaxed + atoms_trajectory

    # transform to adsorption energies
    energies = [atoms.get_potential_energy() for atoms in training_data]
    for atoms, energy in zip(training_data, energies, strict=True):
        atoms.calc = SinglePointCalculator(atoms, energy=energy - (e_surf + e_h))

    write(ROOT_PATH / 'training_data.traj', training_data)

    rss = generate_rss(calc, training_data)

    # transform to adsorption energies
    energies = [atoms.get_potential_energy() for atoms in rss]
    for atoms, energy in zip(rss, energies, strict=True):
        atoms.calc = SinglePointCalculator(atoms, energy=energy - (e_surf + e_h))

    write(ROOT_PATH / 'rss_data.traj', rss)


def generate_kinked_training_data_random_single_point(calc: Calculator) -> list[Atoms]:
    rng = np.random.default_rng(42)

    surface = generate_gold_kinked()
    cell_x, cell_y = surface.cell[0, 0], surface.cell[1, 1]
    zs = np.unique(surface.positions[:, 2])
    z_min, z_max = zs[-4], zs[-1]

    # random coordinates
    n_random = 100
    atoms_list: list[Atoms] = []

    while len(atoms_list) < n_random:
        p = rng.uniform([0, 0, z_min + 0.8], [cell_x, cell_y, z_max + 4.0])
        _, ds = get_distances(surface.positions, [p], surface.cell, surface.pbc)
        if np.min(ds) < 1.0:
            continue
        atoms_list.append(surface + Atoms('H', positions=[p]))

    # evaluate
    for atoms in tqdm(atoms_list):
        atoms.calc = calc
        atoms.calc = SinglePointCalculator(atoms, energy=atoms.get_potential_energy(), forces=atoms.get_forces())

    return atoms_list


def generate_kinked_training_data_relaxed_grid(calc: Calculator) -> tuple[list[Atoms], list[Atoms]]:
    rng = np.random.default_rng(42)

    surface = generate_gold_kinked()
    cell_x, cell_y = surface.cell[0, 0], surface.cell[1, 1]
    zs = np.unique(surface.positions[:, 2])

    grid_spacing = 0.8
    nx = round(cell_x / grid_spacing)
    ny = round(cell_y / grid_spacing)
    xs = np.linspace(0, cell_x, nx, endpoint=False)
    ys = np.linspace(0, cell_y, ny, endpoint=False)

    atoms_list: list[Atoms] = []

    surface_indices = np.flatnonzero(surface.positions[:, 2] >= zs[-3])
    surface_indices = np.sort(np.concatenate((surface_indices, [23, 24, 25])))

    surface_xys = surface.positions[surface_indices, :2]
    surface_xys = np.column_stack((surface_xys, np.zeros(surface_xys.shape[0])))

    for y in ys:
        for x in xs:
            _, ds = get_distances(surface_xys, [[x, y, 0]], cell=surface.cell, pbc=surface.pbc)
            min_idx = np.argmin(ds)
            z = surface.positions[surface_indices[min_idx], 2] + 1.4
            atoms_list.append(surface + Atoms('H', positions=[[x, y, z]]))

    # evaluate
    relaxation_trajectories: list[Atoms] = []
    for atoms in tqdm(atoms_list):
        atoms.calc = calc
        dyn = BFGS(atoms, logfile=None)
        dyn.attach(append_image, opt=dyn, lst=relaxation_trajectories)
        dyn.run(fmax=0.01, steps=1000)
        atoms.calc = SinglePointCalculator(atoms, energy=atoms.get_potential_energy(), forces=atoms.get_forces())

    # deduplicate minima
    _, ds = get_distances([atoms.positions[-1] for atoms in atoms_list], cell=atoms.cell, pbc=atoms.pbc)
    group_indices = fcluster(linkage(squareform(ds, checks=False), method='complete'), t=0.4, criterion='distance')
    groups: dict[list[Atoms]] = defaultdict(list)
    for i, group_index in enumerate(group_indices):
        groups[group_index].append(atoms_list[i])

    atoms_list = [min(group, key=lambda atoms: atoms.get_potential_energy()) for group in groups.values()]

    atoms_list_trajectory: list[Atoms] = []

    # select from relaxation trajectories
    bin_edges = np.linspace(0.2, 1.0, 9)
    e_min = min(atoms.get_potential_energy() for atoms in atoms_list)
    for bin_min, bin_max in zip(bin_edges[:-1], bin_edges[1:], strict=True):
        images = [
            atoms for atoms in relaxation_trajectories if bin_min <= atoms.get_potential_energy() - e_min < bin_max
        ]
        if len(images) > 0:
            selection_indices = rng.choice(len(images), 10)
            atoms_list_trajectory += [images[i] for i in selection_indices]

    return atoms_list, atoms_list_trajectory


def generate_rss(uma_calc: Calculator, training_data: list[Atoms]):
    def generate_initial_random_positions(surface: Atoms, n_positions: int, random_seed: int = 42):
        zs_surface = np.unique(surface.positions[:, 2])
        z_min = zs_surface[-4]
        z_max = zs_surface[-1]

        x_min = 0.0
        x_max = surface.cell[0, 0]
        y_min = 0.0
        y_max = surface.cell[1, 1]

        rng = np.random.default_rng(random_seed)

        positions = []
        while len(positions) < n_positions:
            position = rng.uniform([x_min, y_min, z_min], [x_max, y_max, z_max])

            _, distances = get_distances(surface.positions, [position], surface.cell, surface.pbc)
            if np.min(distances) >= 1.8:
                positions.append(position)

        return positions

    def rss(surface: Atoms, calc: SOAPGPRCalculator, n_positions: int, random_seed: int = 42):
        initial_positions = generate_initial_random_positions(surface, n_positions, random_seed)

        best_energy = np.inf
        best_atoms = None

        all_atoms: list[Atoms] = []

        for position in initial_positions:
            atoms = surface + Atom('H', position=position)
            atoms.calc = calc
            dyn = BFGS(atoms, logfile=None)
            dyn.run(fmax=0.01)

            atoms = copy_atoms(atoms)
            all_atoms.append(atoms)

            if atoms.get_potential_energy() < best_energy:
                best_energy = atoms.get_potential_energy()
                best_atoms = atoms

        return all_atoms, best_atoms

    surface = generate_gold_kinked()

    soap = SOAP(species=['Au', 'H'], r_cut=5.0, n_max=2, l_max=3, periodic=True)

    soaps = soap.create(training_data)
    energies = np.array([atoms.get_potential_energy() for atoms in training_data])

    soaps_train, _, energies_train, _ = train_test_split(
        soaps, energies, train_size=0.8, test_size=0.2, random_state=42
    )

    gpr = SGPR()
    gpr.fit(soaps_train, energies_train)

    calc = SOAPGPRCalculator(soap, gpr)

    rss, _ = rss(surface, calc, n_positions=50)

    for atoms in tqdm(rss):
        atoms.calc = uma_calc
        atoms.calc = SinglePointCalculator(atoms, energy=atoms.get_potential_energy(), forces=atoms.get_forces())

    return rss


def wrap(atoms: Atoms, offset_symbol: str = 'Au', offset: float = 0.5) -> Atoms:
    # wrap while preventing spurious jumping of atoms on the edge of the cell
    results = atoms.calc.results if atoms.calc else None
    atoms = atoms.copy()
    atoms.positions[atoms.symbols == offset_symbol, :2] += offset
    atoms.wrap()
    atoms.positions[atoms.symbols == offset_symbol, :2] -= offset
    if results:
        atoms.calc = SinglePointCalculator(atoms, **results)
    return atoms


def reindex(atoms: Atoms, original: Atoms, reindex_symbol: str = 'Au') -> Atoms:
    results = atoms.calc.results if atoms.calc else None
    indices = np.array([i for i, atom in enumerate(atoms) if atom.symbol == reindex_symbol])
    _, dists = get_distances(original.positions[indices], atoms.positions[indices], cell=atoms.cell, pbc=atoms.pbc)
    row, col = linear_sum_assignment(dists)
    perm = np.arange(len(atoms))
    perm[indices[row]] = indices[col]
    new = atoms[perm]
    if results:
        results = {
            k: (v[perm] if isinstance(v, np.ndarray) and v.shape[:1] == (len(atoms),) else v)
            for k, v in results.items()
        }
        new.calc = SinglePointCalculator(new, **results)
    return new


def copy_atoms(atoms: Atoms) -> Atoms:
    res = atoms.copy()
    res.calc = SinglePointCalculator(res)
    res.calc.results.update(atoms.calc.results)
    return res


def append_image(opt: Optimizer, lst: list[Atoms]) -> None:
    lst.append(copy_atoms(opt.atoms))


def main():
    ROOT_PATH.mkdir(parents=True, exist_ok=True)

    write(ROOT_PATH / 'gold_stepped.traj', generate_gold_stepped())
    generate_hydrogen_on_gold_stepped()

    write(ROOT_PATH / 'gold_kinked.traj', generate_gold_kinked())
    generate_kinked_training_data()

    with tarfile.open(ROOT_PATH.parent / 'ml-data.tar.gz', 'w:gz') as tar:
        for path in ROOT_PATH.iterdir():
            tar.add(path, arcname=path.name)


if __name__ == '__main__':
    main()
