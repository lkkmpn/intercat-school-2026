import traceback
import warnings

import matplotlib
import numpy as np
import torch


def setup_notebook():
    print('setting up notebook, this can take a few seconds...', end='')

    torch.use_deterministic_algorithms(True)

    np.set_printoptions(precision=5, linewidth=96, suppress=True)

    setup_matplotlib()
    protect_matplotlib_against_mace()

    warnings.filterwarnings('ignore', message=r'.+TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD.+', category=UserWarning)

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


def protect_matplotlib_against_mace():
    # https://github.com/ACEsuit/mace/issues/1786

    _orig_setitem = matplotlib.RcParams.__setitem__

    def _protected_setitem(self, key, val):
        if self is matplotlib.rcParams:
            for fs in traceback.extract_stack()[:-1]:
                if '/mace/' in fs.filename:
                    return

        _orig_setitem(self, key, val)

    matplotlib.RcParams.__setitem__ = _protected_setitem
