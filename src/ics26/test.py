import importlib
import importlib.metadata
import shutil
import subprocess
import tempfile
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from packaging.version import Version


@dataclass
class Package:
    distribution_name: str
    import_name: str
    min_version: str


@dataclass
class Check:
    label: str
    check: Callable[[], None]


PACKAGES = [
    Package('ase', 'ase', '3.29.0'),
    Package('ipykernel', 'ipykernel', '6.17.1'),
    Package('ipywidgets', 'ipywidgets', '7.7.1'),
    Package('matplotlib', 'matplotlib', '3.10.0'),
    Package('numpy', 'numpy', '2.1.3'),
    Package('colorspacious', 'colorspacious', '1.1.2'),
    Package('dscribe', 'dscribe', '2.1.2'),
    Package('matplotlib-plus-plus', 'mplpp', '0.0.1'),
    Package('scikit-learn', 'sklearn', '1.6.1'),
    Package('scipy', 'scipy', '1.16.3'),
    Package('torch', 'torch', '2.8.0'),
    Package('tqdm', 'tqdm', '4.67.3'),
    Package('mace-torch', 'mace', '0.3.16'),
    Package('sella', 'sella', '2.6.0'),
]


def check_package(package: Package) -> None:
    importlib.import_module(package.import_name)
    version = importlib.metadata.version(package.distribution_name)
    if Version(version) < Version(package.min_version):
        raise RuntimeError(f'{package.distribution_name} {version} is older than required {package.min_version}')


def check_ase() -> None:
    from ase.build import bulk
    from ase.calculators.emt import EMT

    atoms = bulk('Cu', cubic=True)
    atoms.calc = EMT()
    atoms.get_potential_energy()


def check_ipywidgets():
    from ipywidgets import IntSlider

    IntSlider()


def check_matplotlib() -> None:
    import io

    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use('Agg')
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.plot([0, 1], [0, 1])
    fig.savefig(io.BytesIO(), format='png')
    plt.close(fig)


def check_numpy_scipy() -> None:
    import numpy as np
    from scipy.linalg import solve

    a = np.array([[3.0, 1.0], [1.0, 2.0]])
    assert np.allclose(a @ solve(a, [9.0, 8.0]), [9.0, 8.0])


def check_dscribe() -> None:
    from ase.build import molecule
    from dscribe.descriptors.soap import SOAP

    soap = SOAP(species=['H', 'O'], r_cut=2.0, n_max=2, l_max=2)
    soap.create(molecule('H2O'))


def check_sklearn() -> None:
    import numpy as np
    from sklearn.decomposition import PCA

    PCA(n_components=2).fit_transform(np.random.rand(20, 5))


def check_torch() -> None:
    import torch

    x = torch.randn(4, 4, requires_grad=True)
    (x @ x).sum().backward()


def check_sella():
    from ase.build import molecule
    from ase.calculators.emt import EMT
    from sella import Sella

    atoms = molecule('H2O')
    atoms.calc = EMT()
    Sella(atoms, order=0, internal=True, logfile=None).run(fmax=0.05, steps=100)


def check_gfortran() -> None:
    exe = shutil.which('gfortran')
    if exe is None:
        raise RuntimeError('gfortran not found')
    with tempfile.TemporaryDirectory() as tmpdir:
        src = Path(tmpdir, 'hello.f90')
        src.write_text("program hello\n print *, 'ok'\nend program hello\n")
        out = subprocess.run([exe, str(src), '-o', str(Path(tmpdir, 'hello'))], capture_output=True, text=True)
        if out.returncode != 0:
            raise RuntimeError(f'gfortran failed:\n{out.stderr}')
        out = subprocess.run([str(Path(tmpdir, 'hello'))], capture_output=True, text=True)
        if out.returncode != 0 or 'ok' not in out.stdout:
            raise RuntimeError(f'gfortran exe failed:\n{out.stderr}')


def check_make() -> None:
    exe = shutil.which('make')
    if exe is None:
        raise RuntimeError('make not found')
    with tempfile.TemporaryDirectory() as tmpdir:
        Path(tmpdir, 'Makefile').write_text('all:\n\t@echo ok\n')
        out = subprocess.run([exe, '-C', tmpdir], capture_output=True, text=True)
        if out.returncode != 0 or 'ok' not in out.stdout:
            raise RuntimeError(f'make failed:\n{out.stderr}')


def check_tool(exe_name: str, version_flag: str = '--version') -> None:
    exe = shutil.which(exe_name)
    if exe is None:
        raise RuntimeError(f'{exe_name} not found')
    out = subprocess.run([exe, version_flag], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f'{exe_name} found at {exe} but failed to run:\n{out.stderr}')


CHECKS = [
    Check(f'{package.distribution_name} package', lambda package=package: check_package(package))
    for package in PACKAGES
] + [
    Check('ase run', check_ase),
    Check('ipywidgets run', check_ipywidgets),
    Check('matplotlib run', check_matplotlib),
    Check('numpy scipy run', check_numpy_scipy),
    Check('dscribe run', check_dscribe),
    Check('sklearn run', check_sklearn),
    Check('torch run', check_torch),
    Check('sella run', check_sella),
    Check('gfortran', check_gfortran),
    Check('make', check_make),
    Check('curl', lambda: check_tool('curl')),
    Check('tar', lambda: check_tool('tar')),
]


def run_check(check: Check) -> bool:
    try:
        check.check()
        print(f'✅ {check.label}')
        return True
    except Exception as e:
        msg = f'{type(e).__name__}: {e}\n{traceback.format_exc()}'
        print(f'❌ {check.label}\n{msg}\n')
        return False


def test_notebook():
    n_errors = 0
    for check in CHECKS:
        result = run_check(check)
        if not result:
            n_errors += 1

    if n_errors == 0:
        print("\nEverything succeeded, you're all good!")
    else:
        print(f'\n{n_errors} errors occurred!')
