import matplotlib.pyplot as plt
import numpy as np


def plot_abundances(
    filename: str, plot_species: list[str], reference_abundances: dict[str, float] | None = None
) -> None:
    """Plot the fractional abundances of a list of species from the output of
    an astrochemical model.

    Parameters
    ----------
    filename : str
        Path to file containing the output.
    plot_species : list[str]
        List of species to plot.
    reference_abundances : dict[str, float] | None, optional
        Reference abundances to compare against, keyed by species name.
    """

    all_species, times, abundances = read_abundances(filename)

    fig, ax = plt.subplots()

    for i, species in enumerate(plot_species):
        species_index = all_species.index(species)

        # plot species abundance as function of time as solid line
        ax.plot(times, abundances[:, species_index], c=f'C{i}', label=species)

        # plot reference abundance as dotted line
        if reference_abundances and species in reference_abundances:
            ax.axhline(reference_abundances[species], c=f'C{i}', ls=':')

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlim(np.min(times), np.max(times))

    if np.min(abundances) < 1e-12:
        ax.set_ylim(bottom=1e-12)

    ax.set_xlabel('time (yr)')
    ax.set_ylabel(r'$n(X) / n(\mathrm{H})$')

    ax.legend()

    plt.show()
    plt.close()


def read_abundances(filename: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    with open(filename) as f:
        lines = f.readlines()

    # read list of species
    all_species: list[str] = []

    line_index = 15
    while lines[line_index].strip():
        all_species += lines[line_index].split()
        line_index += 1

    # read abundances as function of time
    # first, read how many time steps exist
    while 'TIME' not in lines[line_index]:  # search for header of first data block
        line_index += 1

    first_abundances_line = line_index
    line_index += 1

    while 'TIME' not in lines[line_index]:  # search for footer of first data block
        line_index += 1

    n_time_steps = line_index - first_abundances_line - 1

    abundances = np.empty((n_time_steps, 0))

    # next, find every data block
    line_index = first_abundances_line

    while True:
        # search for header
        while line_index < len(lines) and 'TIME' not in lines[line_index]:
            line_index += 1

        if line_index >= len(lines):
            break

        # extract data block
        data_block_lines = lines[line_index + 1 : line_index + 1 + n_time_steps]
        data_block = np.loadtxt(data_block_lines)

        times = data_block[:, 0]
        abundances = np.concatenate((abundances, data_block[:, 1:]), axis=1)

        line_index += n_time_steps + 2  # skip past footer

    return all_species, times, abundances


if __name__ == '__main__':
    read_abundances('dc.out')
