import timeit

from ase.atoms import Atoms
from ase.calculators.singlepoint import SinglePointCalculator


def copy_atoms(atoms: Atoms) -> Atoms:
    a = atoms.copy()
    if atoms.calc is not None:
        a.calc = SinglePointCalculator(a, **atoms.calc.results)
    return a


class ImageRecorder:
    def __init__(self, atoms: Atoms):
        self.atoms = atoms
        self.images: list[Atoms] = []

    def __call__(self):
        self.images.append(copy_atoms(self.atoms))


class Timer:
    def __init__(self):
        self._ran = False

    def __enter__(self):
        self.start_time = timeit.default_timer()
        return self

    def __exit__(self, *_):
        self.end_time = timeit.default_timer()
        self._ran = True

    @property
    def runtime(self) -> float | None:
        if not self._ran:
            return None
        return self.end_time - self.start_time

    def __repr__(self):
        if not self._ran:
            return 'Timer()'
        return f'Timer({self.runtime:.3f})'

    def __str__(self):
        if not self._ran:
            return '<not ran>'
        return f'{self.runtime:.3f}'
