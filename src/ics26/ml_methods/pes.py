import numpy as np
from ase.atoms import Atoms
from ase.constraints.fixed_line import FixedLine
from ase.geometry import get_distances, wrap_positions
from ase.optimize.bfgs import BFGS
from dscribe.descriptors.soap import SOAP
from tqdm import tqdm

from ics26.ml_methods.gpr import GPR, SGPR, SOAPGPRCalculator


def get_sphere_points(n: int) -> np.ndarray:
    golden_ratio = (1 + np.sqrt(5)) / 2
    angle_increment = 2 * np.pi * golden_ratio

    i = np.arange(n)
    distance = i / n
    incline = np.acos(1 - 2 * distance)
    azimuth = angle_increment * i

    xs = np.sin(incline) * np.cos(azimuth)
    ys = np.sin(incline) * np.sin(azimuth)
    zs = np.cos(incline)

    return np.column_stack((xs, ys, zs))


def get_pes_points(
    atoms: Atoms, *, radius: float, indices: list[int], z_threshold: float, n_sphere_points: int
) -> np.ndarray:
    sphere_points = get_sphere_points(n_sphere_points)

    all_points = np.concatenate([sphere_points * radius + atoms.positions[index] for index in indices], axis=0)
    origin_indices = np.repeat(indices, n_sphere_points)

    mask = all_points[:, 2] >= z_threshold
    all_points = all_points[mask]
    origin_indices = origin_indices[mask]

    _, ds = get_distances(atoms.positions, all_points, cell=atoms.cell, pbc=atoms.pbc)

    closest_atoms_indices = np.argmin(ds, axis=0)
    points = all_points[closest_atoms_indices == origin_indices]

    points = wrap_positions(points, cell=atoms.cell)

    return points


def evaluate_pes_points(surface: Atoms, symbol: str, points: np.ndarray, gpr: GPR | SGPR, soap: SOAP) -> np.ndarray:
    atoms_list = [surface + Atoms(symbol, positions=[p]) for p in points]
    soap_matrices = soap.create(atoms_list)

    energies = np.array([gpr.predict_energy(x) for x in soap_matrices])
    return energies


def get_start_z(surface: Atoms, xy: np.ndarray, d_target: float = 2.2, dz: float = 0.1) -> float:
    z = np.max(surface.positions[:, 2]) + 3.0

    while z > np.min(surface.positions[:, 2]):
        _, ds = get_distances(surface.positions, [*xy, z], cell=surface.cell, pbc=surface.pbc)
        if np.min(ds) <= d_target:
            return z
        z -= dz

    raise ValueError('No start z found')


def evaluate_pes_point(surface: Atoms, symbol: str, xy: np.ndarray, calc: SOAPGPRCalculator) -> float:
    pos = [*xy, get_start_z(surface, xy)]

    atoms: Atoms = surface + Atoms('H', positions=[pos])
    atoms.set_constraint([*atoms.constraints, FixedLine([len(atoms) - 1], [0, 0, 1])])

    atoms.calc = calc
    BFGS(atoms, logfile=None, maxstep=0.1).run(fmax=0.02, steps=100)

    return atoms.get_potential_energy()


def evaluate_pes_points_grid(surface: Atoms, symbol: str, gpr: GPR, soap: SOAP):
    calc = SOAPGPRCalculator(soap, gpr)

    xs = np.linspace(0, 1, 21)
    ys = np.linspace(0, 1, 21)
    xx, yy = np.meshgrid(xs, ys)

    zz = np.empty(xx.shape)

    for i, y in enumerate(tqdm(ys)):
        for j, x in enumerate(xs):
            xy = [x, y] @ surface.cell[:2, :2]
            zz[i, j] = evaluate_pes_point(surface, symbol, xy, calc)

    return xx, yy, zz
