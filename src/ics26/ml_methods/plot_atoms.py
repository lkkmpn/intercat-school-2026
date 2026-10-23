import itertools
import warnings
from typing import Any

import ase.data
import ase.data.colors
import numpy as np
from ase.atoms import Atoms
from colorspacious import cspace_convert
from matplotlib.axes import Axes
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle
from mplpp.atoms import get_atoms_limits, plane_to_indices
from mplpp.limits import Limits
from mplpp.utils import disable_ticks


def _plot_atoms(
    ax: Axes,
    atoms: Atoms,
    *,
    plane: str = 'xy',
    radius_factor: float = 1.0,
    limits_indices: bool | list[int] | None = None,
    equal_aspect: bool = True,
    no_ticks: bool = True,
    patch_limits: Limits | None = None,
    patch_kwargs: dict[str, Any] = None,
    repeat: bool = False,
    repeat_patch_kwargs: dict[str, Any] = None,
    collection_kwargs: dict[str, Any] = None,
    darken: bool = False,
    darken_threshold: float = np.nan,
    darken_factor: float = 0.1,
    lighten: bool = False,
    lighten_threshold: float = np.nan,
    lighten_factor: float = 0.1,
    repeat_desaturate: float = 0.0,
    only_return: bool = False,
) -> None:
    default_patch_kwargs = {'ec': 'k'}

    if patch_kwargs is None:
        patch_kwargs = {}
    if repeat_patch_kwargs is None:
        repeat_patch_kwargs = {}
    if collection_kwargs is None:
        collection_kwargs = {}
    if darken and lighten:
        raise ValueError('`darken` and `lighten` cannot be used at the same time.')

    d0, d1, d2, kv = plane_to_indices(plane)

    radii = radius_factor * ase.data.covalent_radii[atoms.numbers]
    colors = ase.data.colors.jmol_colors[atoms.numbers]

    if patch_limits:
        patch_limits = patch_limits.copy()
        patch_limits.add_margin(0.02)  # buffer

    if limits_indices is not False:
        if limits_indices is None:
            limits_indices = range(len(atoms))
        limits_indices = list(limits_indices)

        limits = get_atoms_limits(atoms[limits_indices], plane=plane, radius_factor=radius_factor)
        limits.add_margin(0.05)
        limits.apply_to_axes(ax)

    if equal_aspect:
        ax.set_aspect('equal')

    if no_ticks:
        disable_ticks(ax)

    if repeat:
        atoms_ = atoms.copy()
        pbc = atoms.get_pbc()

        if patch_limits:
            repeat_limits = patch_limits
        elif limits_indices is not False:
            repeat_limits = limits
        else:
            repeat_limits = Limits.from_axes(ax)
            if repeat_limits == (0.0, 1.0, 0.0, 1.0):
                warnings.warn(
                    'No limits provided or set on the axes, `repeat=True` might give unexpected results.', stacklevel=2
                )

        # atoms can extend past their cell a bit
        # ugly fix, would be nicer if this gets replaced with something better
        repeat_limits.add_margin(0.1)

        extrema_uc = np.zeros((4, 3))
        for i, extremum in enumerate(itertools.product(repeat_limits.get_xlim(), repeat_limits.get_ylim())):
            coord = np.zeros(3)
            coord[[d0, d1]] = list(extremum)
            extrema_uc[i, :] = atoms.cell.scaled_positions(coord)

        repeat_start = np.floor(np.min(extrema_uc[:, [d0, d1]], axis=0)).astype(int)
        repeat_stop = np.ceil(np.max(extrema_uc[:, [d0, d1]], axis=0)).astype(int)

        repeat_range_d0 = range(repeat_start[0], repeat_stop[0]) if pbc[d0] else [0]
        repeat_range_d1 = range(repeat_start[1], repeat_stop[1]) if pbc[d1] else [0]

        for x, y in itertools.product(repeat_range_d0, repeat_range_d1):
            if x == y == 0:
                continue
            offset_uc = np.zeros(3)
            offset_uc[[d0, d1]] = [x, y]
            offset = atoms.cell.cartesian_positions(offset_uc)
            repeat_atoms = atoms.copy()
            repeat_atoms.translate(offset)
            atoms_ += repeat_atoms
    else:
        atoms_ = atoms

    n_atoms = len(atoms)
    patches: list[Circle] = []

    for i in np.argsort(kv * atoms_.positions[:, d2]):
        patch_kwargs_ = (
            {**default_patch_kwargs, **patch_kwargs}
            if i < n_atoms
            else {**default_patch_kwargs, **patch_kwargs, **repeat_patch_kwargs}
        )

        position = atoms_[i].position
        radius = radii[i % n_atoms]
        color = colors[i % n_atoms]

        if darken:
            if np.isnan(darken_threshold):
                darken_factor_ = darken_factor
            else:
                darken_factor_ = (darken_threshold - kv * position[d2]) * darken_factor
            color = color * min(1, 1 - min(1, darken_factor_))

        if lighten:
            if np.isnan(lighten_threshold):
                lighten_factor_ = lighten_factor
            else:
                lighten_factor_ = (lighten_threshold - kv * position[d2]) * lighten_factor
            color = 1 - (1 - color) * min(1, 1 - min(1, lighten_factor_))

        if repeat_desaturate and i >= n_atoms:
            color = desaturate_cheap(color, repeat_desaturate)

        if patch_limits:
            if (
                position[d0] + radius < patch_limits.xmin
                or position[d0] - radius > patch_limits.xmax
                or position[d1] + radius < patch_limits.ymin
                or position[d1] - radius > patch_limits.ymax
            ):
                continue

        circle = Circle(position[[d0, d1]], radius=radius, fc=color, **patch_kwargs_)
        patches.append(circle)

    p = PatchCollection(patches, match_original=True, **collection_kwargs)

    if not only_return:
        ax.add_collection(p, autolim=False)

    return p


def desaturate(rgb, amount=0.5):
    jab = cspace_convert(rgb, 'sRGB1', 'CAM02-UCS')
    jab[..., 1:] *= 1 - amount
    return np.clip(cspace_convert(jab, 'CAM02-UCS', 'sRGB1'), 0, 1)


def desaturate_cheap(rgb, amount=0.5):
    rgb = np.asarray(rgb, dtype=float)
    gray = rgb @ np.array([0.2126, 0.7152, 0.0722])  # luminance (approx., in sRGB)
    return gray[..., None] + (rgb - gray[..., None]) * (1 - amount)
