import itertools
import lzma
import tarfile
from pathlib import Path

import numpy as np
from ase.atoms import Atoms
from tqdm.contrib.concurrent import process_map

from ics26.ml_methods.plot import _get_limits_xy, _plot_atoms_slider_figure


def repeat_args(*args):
    return (itertools.repeat(arg) for arg in args)


def prerender_slider(atoms_list: list[Atoms], output_filename: str):
    atoms = atoms_list[0]  # assumed that all are same surface

    darken_threshold = np.max(atoms.positions[atoms.symbols == 'Au', 2])

    limits = _get_limits_xy(atoms)
    aspect = limits.get_width() / limits.get_height()

    height = 4.0
    width = height * aspect

    output_path = Path(output_filename)
    output_path.unlink(missing_ok=True)

    bufs = process_map(
        _plot_atoms_slider_figure,
        atoms_list,
        *repeat_args(width, height, limits, darken_threshold),
        desc='Preparing visualization',
        leave=False,
    )

    with tarfile.open(output_path, 'x:xz', preset=9 | lzma.PRESET_EXTREME) as tar:
        for i, buf in enumerate(bufs):
            info = tarfile.TarInfo(name=f'{i:04d}.png')
            info.size = buf.getbuffer().nbytes
            buf.seek(0)
            tar.addfile(info, buf)
