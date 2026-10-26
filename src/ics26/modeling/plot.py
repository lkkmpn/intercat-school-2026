import io
import itertools
import operator
import warnings
from functools import reduce

import ipywidgets
import matplotlib.pyplot as plt
from ase.atoms import Atoms
from IPython.display import display
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from mplpp.atoms import get_atoms_limits, plane_to_indices
from mplpp.limits import Limits
from tqdm.contrib.concurrent import process_map

from ics26.modeling.plot_atoms import _plot_atoms


def repeat_args(*args):
    return (itertools.repeat(arg) for arg in args)


def plot_atoms(atoms: Atoms, *, plane: str = 'xy', show_indices: bool = False):
    limits = get_atoms_limits(atoms, plane=plane, radius_factor=0.9)
    limits.add_margin(0.01)

    fig, ax = plt.subplots(figsize=(3, 3))

    _plot_atoms(ax, atoms, plane=plane, radius_factor=0.9, patch_limits=limits, limits_indices=False)

    limits.apply_to_axes(ax)

    if show_indices:
        d0, d1, _, _ = plane_to_indices(plane)

        for i, atom in enumerate(atoms):
            p = atom.position[[d0, d1]]
            ax.annotate(f'{i}', xy=p, ha='center', va='center')

    ax.set_axis_off()


def _plot_atoms_slider_figure(atoms: Atoms, width: float, height: float, plane: str, limits: Limits):
    fig = Figure(figsize=(width, height))
    FigureCanvasAgg(fig)

    ax = fig.subplots()

    _plot_atoms(ax, atoms, plane=plane, radius_factor=0.9, patch_limits=limits, limits_indices=False)

    limits.apply_to_axes(ax)

    ax.set_axis_off()

    title_parts: list[str] = []

    if atoms.info.get('label'):
        title_parts.append(atoms.info.get('label'))

    if atoms.calc is not None and 'energy' in atoms.calc.results:
        energy = atoms.calc.results.get('energy')
        title_parts.append(rf'$E = {energy:.2f}\,\mathrm{{eV}}$')

    title = ' · '.join(title_parts)
    if title == '':
        title = '\u200b'

    ax.set_title(title, loc='left')

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pil_kwargs={'compress_level': 0})

    return buf


def plot_atoms_slider(atoms_list: list[Atoms], plane: str = 'xy'):
    all_limits = [get_atoms_limits(atoms, plane=plane, radius_factor=0.9) for atoms in atoms_list]

    limits = reduce(operator.or_, all_limits)
    limits.add_margin(0.01)
    aspect = limits.get_width() / limits.get_height()

    height = 3.0
    width = height * aspect

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)

        bufs = process_map(
            _plot_atoms_slider_figure,
            atoms_list,
            *repeat_args(width, height, plane, limits),
            desc='Preparing visualization',
            leave=False,
        )

    slider = ipywidgets.SelectionSlider(
        options=[(f'{i}', i) for i in range(len(atoms_list))],
        value=0,
        description='Image index:',
        layout=ipywidgets.Layout(width='400px'),
    )
    image = ipywidgets.Image(format='png')

    def render(_=None):
        image.value = bufs[slider.value].getvalue()

    slider.observe(render, names='value')
    render()
    display(slider, image)
