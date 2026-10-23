import tarfile
import urllib.request

import matplotlib
import numpy as np

FILE_URL = 'https://github.com/lkkmpn/intercat-school-2026/raw/refs/heads/main/files/ml-data.tar.gz'


def setup_notebook():
    print('setting up notebook, this can take a few seconds...', end='')

    np.set_printoptions(precision=4, linewidth=96, suppress=True)

    setup_matplotlib()
    download_files()
    import_modules()
    cache_cacheables()

    print(' done!')


def setup_matplotlib():
    matplotlib.rcParams.update(
        {
            'axes.titlesize': 'medium',
            'figure.subplot.left': 0.0,
            'figure.subplot.right': 1.0,
            'figure.subplot.bottom': 0.0,
            'figure.subplot.top': 1.0,
        }
    )


def download_files():
    with urllib.request.urlopen(FILE_URL) as r:
        with tarfile.open(fileobj=r, mode='r|gz') as tar:
            tar.extractall(path='.')


def import_modules():
    from ics26.ml_methods import dimensionality_reduction, gpr, pes, plot  # noqa: F401


def cache_cacheables():
    from ics26.ml_methods.dimensionality_reduction import atoms_for_setup
    from ics26.ml_methods.plot import _get_span, _prerendered_sliders

    atoms_for_setup()
    _get_span()
    _prerendered_sliders()
