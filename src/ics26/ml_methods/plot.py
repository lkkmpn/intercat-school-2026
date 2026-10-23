import io
import itertools
import tarfile
from dataclasses import dataclass
from functools import cache

import ipywidgets
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from ase.atoms import Atoms
from ase.io import read
from IPython.display import display
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.lines import AxLine
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.axes_grid import ImageGrid
from mplpp.atoms import get_atoms_limits, plot_cell
from mplpp.limits import Limits
from mplpp.utils import disable_ticks
from scipy.interpolate import RegularGridInterpolator
from tqdm.contrib.concurrent import process_map

from ics26.ml_methods.dimensionality_reduction import Reduction
from ics26.ml_methods.plot_atoms import _plot_atoms


@cache
def _get_span() -> tuple[float, float]:
    atoms = read('gold_stepped.traj')
    limits = get_atoms_limits(atoms, plane='xy', radius_factor=1.0)
    limits.add_margin(0.25)
    span = max(limits.get_width(), limits.get_height())
    return (span, span)


def _get_limits_xy(atoms: Atoms) -> Limits:
    span = _get_span()
    return Limits.from_center(get_atoms_limits(atoms, plane='xy', radius_factor=1.0).get_center(), *span)


def repeat_args(*args):
    return (itertools.repeat(arg) for arg in args)


@dataclass
class PrerenderedSlider:
    atoms_list: list[Atoms]
    bufs: list[io.BytesIO]


@cache
def _prerendered_sliders() -> list[PrerenderedSlider]:
    return [
        PrerenderedSlider(
            read('gold_stepped_hydrogen_x_diffusion.traj', index=':'),
            _prerendered_slider('gold_stepped_hydrogen_x_diffusion.tar.xz'),
        ),
        PrerenderedSlider(
            sorted(read('training_data.traj', index=':'), key=lambda atoms: atoms.get_potential_energy()),
            _prerendered_slider('training_data.tar.xz'),
        ),
    ]


def _prerendered_slider(path: str) -> list[io.BytesIO]:
    with tarfile.open(path, 'r:xz') as tar:
        members = [m for m in tar.getmembers() if m.isfile() and m.name.endswith('.png')]
        members.sort(key=lambda m: m.name)
        return [io.BytesIO(tar.extractfile(m).read()) for m in members]


def match_prerendered_slider(atoms_list: list[Atoms]) -> list[io.BytesIO] | None:
    for prerendered_slider in _prerendered_sliders():
        if len(atoms_list) == len(prerendered_slider.atoms_list) and all(
            atoms == prerendered_atoms
            for atoms, prerendered_atoms in zip(atoms_list, prerendered_slider.atoms_list, strict=True)
        ):
            return prerendered_slider.bufs

    return None


def plot_atoms(
    atoms_list: Atoms | list[Atoms],
    *,
    plane: str = 'xy',
    darken_threshold_symbol: str = 'Au',
    repeat_desaturate: float = 0.8,
):
    if isinstance(atoms_list, Atoms):
        atoms_list = [atoms_list]

    atoms = atoms_list[0]  # assumed that all are same surface

    if plane == 'xy':
        darken = True
        darken_threshold = np.max(atoms.positions[atoms.symbols == darken_threshold_symbol, 2])
        darken_factor = 0.2
    else:
        darken = False
        darken_threshold = np.nan
        darken_factor = 0.0

    if plane == 'xy':
        limits = _get_limits_xy(atoms)
    else:
        limits = get_atoms_limits(atoms, plane=plane, radius_factor=1.0)
        limits.add_margin(0.25)

    aspect = limits.get_width() / limits.get_height()

    height = 4.0
    axes_pad = 0.1
    width = height * aspect * len(atoms_list) + axes_pad * (len(atoms_list) - 1)

    fig = plt.figure(figsize=(width, height))
    grid = ImageGrid(fig, 111, (1, len(atoms_list)), axes_pad=axes_pad, aspect=True)

    for ax, atoms in zip(grid, atoms_list, strict=True):
        _plot_atoms(
            ax,
            atoms,
            plane=plane,
            radius_factor=1.0,
            patch_limits=limits,
            repeat=True,
            repeat_patch_kwargs={'ls': ':'},
            darken=darken,
            darken_threshold=darken_threshold,
            darken_factor=darken_factor,
            repeat_desaturate=repeat_desaturate,
        )

        plot_cell(ax, atoms.cell, plane=plane, collection_kwargs={'edgecolors': 'C3'})

        limits.apply_to_axes(ax)

        if atoms.info.get('label'):
            ax.set_title(atoms.info.get('label'), loc='left')


def _plot_atoms_slider_figure(atoms: Atoms, width: float, height: float, limits: Limits, darken_threshold: float):
    fig = Figure(figsize=(width, height))
    FigureCanvasAgg(fig)

    ax = fig.subplots()

    _plot_atoms(
        ax,
        atoms,
        plane='xy',
        radius_factor=1.0,
        patch_limits=limits,
        limits_indices=False,
        repeat=True,
        repeat_patch_kwargs={'ls': ':'},
        darken=True,
        darken_threshold=darken_threshold,
        darken_factor=0.2,
        repeat_desaturate=0.8,
    )

    plot_cell(ax, atoms.cell, plane='xy', collection_kwargs={'edgecolors': 'C3'})

    limits.apply_to_axes(ax)

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


def plot_atoms_slider(atoms_list: list[Atoms], sorted_by_energy: bool = False):
    atoms = atoms_list[0]  # assumed that all are same surface

    darken_threshold = np.max(atoms.positions[atoms.symbols == 'Au', 2])

    limits = _get_limits_xy(atoms)
    aspect = limits.get_width() / limits.get_height()

    height = 4.0
    width = height * aspect

    if sorted_by_energy:
        atoms_list = sorted(atoms_list, key=lambda atoms: atoms.get_potential_energy())

    bufs = match_prerendered_slider(atoms_list) or process_map(
        _plot_atoms_slider_figure,
        atoms_list,
        *repeat_args(width, height, limits, darken_threshold),
        desc='Preparing visualization',
        leave=False,
    )

    labels = [atoms.info.get('label') for atoms in atoms_list]
    options = [(f'{i} ({label})' if label else f'{i}', i) for i, label in enumerate(labels)]

    slider = ipywidgets.SelectionSlider(
        options=options, value=0, description='Image index:', layout=ipywidgets.Layout(width='400px')
    )
    image = ipywidgets.Image(format='png')

    def render(_=None):
        image.value = bufs[slider.value].getvalue()

    slider.observe(render, names='value')
    render()
    display(slider, image)


@dataclass
class QuiverBatch:
    vectors: np.ndarray
    kwargs: dict
    labels: list[str] | None


def _plot_reduction(ax: Axes, reduction: Reduction, quiver_batches: list[QuiverBatch]):
    default_kwargs = {'angles': 'xy', 'scale_units': 'xy', 'scale': 1, 'zorder': 3}

    ax.update_datalim([(reduction.xmin, reduction.ymin), (reduction.xmax, reduction.ymax)])
    ax.spines[['left', 'bottom']].set_position('zero')
    ax.spines[['top', 'right']].set_visible(False)

    for batch in quiver_batches:
        n = batch.vectors.shape[0]
        X = Y = [0] * n
        U, V = batch.vectors.T

        ax.quiver(X, Y, U, V, **(default_kwargs | batch.kwargs))

        labels = batch.labels if batch.labels is not None else []

        for u, v, label in zip(U, V, labels, strict=False):
            p0 = ax.transData.transform((0, 0))
            p1 = ax.transData.transform((u, v))
            d = p1 - p0
            norm = np.hypot(*d)
            if norm == 0:
                off = np.array([1.0, 0.0])
            else:
                off = d / norm
            offset = off * 72.0 * 0.2
            ax.annotate(
                label,
                xy=(u, v),
                xytext=offset,
                textcoords='offset points',
                ha='center',
                va='center',
                bbox={'boxstyle': 'round', 'facecolor': (1.0, 1.0, 1.0, 0.5), 'pad': 0.3},
            )

    ax.annotate(
        'PC1',
        xy=(1, 0),
        xytext=(4, 0),
        xycoords=ax.get_yaxis_transform(),
        textcoords='offset points',
        ha='left',
        va='center',
    )
    ax.annotate(
        'PC2',
        xy=(0, 1),
        xytext=(0, 4),
        xycoords=ax.get_xaxis_transform(),
        textcoords='offset points',
        ha='center',
        va='bottom',
    )


def plot_reduction(reduction: Reduction, transformed_vectors: np.ndarray, labels: list[str] | None = None):
    fig, ax = plt.subplots(figsize=(4.0, 4.0))

    if transformed_vectors.ndim == 1:
        transformed_vectors = transformed_vectors.reshape(1, -1)

    n = transformed_vectors.shape[0]

    _plot_reduction(ax, reduction, [QuiverBatch(transformed_vectors, {'color': [f'C{i}' for i in range(n)]}, labels)])


def _plot_reduction_slider_figure(
    atoms: Atoms,
    index: int,
    height: float,
    reduction: Reduction,
    transformed_vectors: np.ndarray,
    reference_vectors: np.ndarray,
    reference_labels: list[str],
):
    fig, ax = plt.subplots(figsize=(height, height))
    FigureCanvasAgg(fig)

    _plot_reduction(
        ax,
        reduction,
        [
            QuiverBatch(
                reference_vectors, {'color': [f'C{i}' for i in range(reference_vectors.shape[0])]}, reference_labels
            ),
            QuiverBatch(transformed_vectors[index].reshape(1, -1), {'color': 'k', 'width': 0.005}, None),
        ],
    )

    ax.plot(transformed_vectors[: index + 1, 0], transformed_vectors[: index + 1, 1], c='k', lw=0.8, ls=':')

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')

    return buf


def plot_atoms_and_reduction_slider(
    atoms_list: list[Atoms],
    reduction: Reduction,
    transformed_vectors: np.ndarray,
    *,
    reference_vectors: np.ndarray | None = None,
    reference_labels: list[str] | None = None,
):
    reference_vectors = reference_vectors if reference_vectors is not None else np.array([])
    reference_labels = reference_labels if reference_labels is not None else []

    if reference_vectors.ndim == 1:
        reference_vectors = reference_vectors.reshape(1, -1)

    atoms = atoms_list[0]  # assumed that all are same surface

    darken_threshold = np.max(atoms.positions[atoms.symbols == 'Au', 2])

    limits = _get_limits_xy(atoms)
    aspect = limits.get_width() / limits.get_height()

    height = 4.0
    width = height * aspect

    atoms_bufs = match_prerendered_slider(atoms_list) or process_map(
        _plot_atoms_slider_figure,
        atoms_list,
        *repeat_args(width, height, limits, darken_threshold),
        desc='Preparing visualization',
        leave=False,
    )

    reduction_bufs = process_map(
        _plot_reduction_slider_figure,
        atoms_list,
        range(len(atoms_list)),
        *repeat_args(height, reduction, transformed_vectors, reference_vectors, reference_labels),
        desc='Preparing visualization',
        leave=False,
    )

    labels = [atoms.info.get('label') for atoms in atoms_list]
    options = [(f'{i} ({label})' if label else f'{i}', i) for i, label in enumerate(labels)]

    slider = ipywidgets.SelectionSlider(
        options=options, value=0, description='Image index:', layout=ipywidgets.Layout(width='400px')
    )
    atoms_image = ipywidgets.Image(format='png')
    reduction_image = ipywidgets.Image(format='png')
    images = ipywidgets.HBox([atoms_image, reduction_image], layout=ipywidgets.Layout(align_items='center'))

    def render(_=None):
        atoms_image.value = atoms_bufs[slider.value].getvalue()
        reduction_image.value = reduction_bufs[slider.value].getvalue()

    slider.observe(render, names='value')
    render()
    display(slider, images)


def plot_parity(ax: Axes, targets: np.ndarray, predictions: np.ndarray, title: str = ''):
    ax.scatter(targets, predictions, s=9)

    ax.set_xlabel(r'$E_{\mathrm{target}}$ (eV)')
    ax.set_ylabel(r'$E_{\mathrm{predicted}}$ (eV)')

    ax.set_title(title, loc='left')

    # make square
    ax.set_aspect('equal')

    min_val = np.nanmin(np.concatenate((targets, predictions)))
    max_val = np.nanmax(np.concatenate((targets, predictions)))

    ax.update_datalim([(min_val, min_val), (max_val, max_val)])

    # parity line
    line = AxLine(xy1=(0, 0), xy2=None, slope=1.0, color='k')
    ax.add_line(line)


def plot_potential_energy_surface(atoms: Atoms, xx: np.ndarray, yy: np.ndarray, zz: np.ndarray):
    limits = _get_limits_xy(atoms)
    aspect = limits.get_width() / limits.get_height()

    height = 4.0
    width = height * aspect

    fig, ax = plt.subplots(figsize=(width, height))

    # render atoms to buffer, with inverted luminance as alpha
    bbox = ax.get_window_extent()

    fig_atoms = Figure(figsize=(int(bbox.width) / fig.dpi, int(bbox.height) / fig.dpi))
    canvas = FigureCanvasAgg(fig_atoms)
    ax_atoms = fig_atoms.add_axes([0, 0, 1, 1])
    ax_atoms.axis('off')

    _plot_atoms(
        ax_atoms,
        atoms,
        plane='xy',
        radius_factor=1.0,
        patch_limits=limits,
        limits_indices=False,
        repeat=True,
        repeat_patch_kwargs={'ls': ':'},
        lighten=True,
        lighten_factor=1.0,
    )

    plot_cell(ax_atoms, atoms.cell, plane='xy', collection_kwargs={'edgecolors': 'k', 'linewidths': 0.4})

    limits.apply_to_axes(ax_atoms)

    canvas.draw()
    rgb = np.asarray(canvas.buffer_rgba())[..., :3] / 255
    luminance = np.mean(rgb, axis=-1)
    overlay = np.zeros((*luminance.shape, 4))
    overlay[..., 3] = (1 - luminance) * 0.5

    # plot rest
    xs, ys = xx[0, :], yy[:, 0]

    Lx = xs[-1] - xs[0] + (xs[1] - xs[0])
    Ly = ys[-1] - ys[0] + (ys[1] - ys[0])

    offsets = (-1, 0, 1)

    zzt = np.tile(zz, (len(offsets), len(offsets)))
    xst = np.concatenate([xs + k * Lx for k in offsets])
    yst = np.concatenate([ys + k * Ly for k in offsets])

    interp = RegularGridInterpolator((yst, xst), zzt, method='cubic')

    xf = np.linspace(np.min(xst), np.max(xst), 400)
    yf = np.linspace(np.min(yst), np.max(yst), 400)
    xxf, yyf = np.meshgrid(xf, yf)
    zzf = interp((yyf, xxf))

    cmap = matplotlib.colormaps.get_cmap('PiYG_r')
    norm = TwoSlopeNorm(
        vcenter=0.0, vmin=np.minimum(np.min(zzf), np.nextafter(0, -1)), vmax=np.maximum(np.max(zzf), np.nextafter(0, 1))
    )

    ax.imshow(
        overlay,
        extent=[*limits.get_xlim(), *limits.get_ylim()],
        origin='upper',
        aspect='auto',
        zorder=3,
        interpolation='nearest',
    )

    cax = fig.add_axes([1.05, 0.0, 0.05, 1.0])

    c = ax.contourf(xxf, yyf, zzf, cmap=cmap, norm=norm, levels=255)
    fig.colorbar(c, cax=cax)

    cax.yaxis.set_major_locator(MaxNLocator(nbins=8))
    cax.set_ylabel('Adsorption energy (eV)')

    limits.apply_to_axes(ax)
    disable_ticks(ax)
