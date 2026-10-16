from functools import cache

import numpy as np
from ase.atoms import Atoms
from ase.io import read
from dscribe.descriptors.soap import SOAP
from sklearn.decomposition import PCA


class Reduction:
    def __init__(self, pca: PCA, transformed_input: np.ndarray):
        self.pca = pca

        self.xmin = np.min(transformed_input[:, 0])
        self.xmax = np.max(transformed_input[:, 0])
        self.ymin = np.min(transformed_input[:, 1])
        self.ymax = np.max(transformed_input[:, 1])

    def transform(self, soap_vectors: np.ndarray) -> np.ndarray:
        is_single = soap_vectors.ndim == 1

        if is_single:
            soap_vectors = soap_vectors.reshape(1, -1)

        transformed = self.pca.transform(soap_vectors)

        if is_single:
            transformed = transformed[0]

        return transformed


@cache
def atoms_for_setup() -> list[Atoms]:
    images = read('gold_stepped_hydrogen_x_diffusion.traj', index='0:64')
    return images


def setup_reduction(soap: SOAP) -> Reduction:
    images = atoms_for_setup()
    soap_vectors = soap.create(images, [[-1]] * len(images), n_jobs=-1)[:, 0, :]

    pca = PCA(n_components=2, random_state=42)
    transformed = pca.fit_transform(soap_vectors)

    return Reduction(pca, transformed)


def main():
    soap = SOAP(r_cut=5, n_max=2, l_max=3, species=['Au', 'H'])
    setup_reduction(soap)
    breakpoint()


if __name__ == '__main__':
    main()
