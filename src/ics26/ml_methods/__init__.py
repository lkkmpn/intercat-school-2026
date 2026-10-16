import matplotlib
import numpy as np


def setup_notebook():
    np.set_printoptions(precision=4, linewidth=96, suppress=True)

    matplotlib.rcParams.update(
        {
            'axes.titlesize': 'medium',
            'figure.subplot.left': 0.0,
            'figure.subplot.right': 1.0,
            'figure.subplot.bottom': 0.0,
            'figure.subplot.top': 1.0,
        }
    )
