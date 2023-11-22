import torch


class System:
    """A single system for which we want to run a calculation."""

    def __init__(
        self,
        species: torch.Tensor,
        positions: torch.Tensor,
        cell: torch.Tensor,
    ):
        """
        :param species: species of the atoms/particles in this system. This should
            be a 1D array of integer containing different values for different
            system. The species will typically match the atomic element, but does
            not have to.
        :param positions: positions of the atoms/particles in this system. This
            should be a ``len(species) x 3`` 2D array containing the positions of
            each atom.
        :param cell: 3x3 cell matrix for periodic boundary conditions, where each
            row is one of the cell vector. Use a matrix filled with ``0`` for
            non-periodic systems.
        """

        self._species = species
        self._positions = positions
        self._cell = cell

    @property
    def species(self) -> torch.Tensor:
        """the species of the atoms/particles in this system"""

        return self._species

    @property
    def positions(self) -> torch.Tensor:
        """the positions of the atoms/particles in this system"""

        return self._positions

    @property
    def cell(self) -> torch.Tensor:
        """
        the bounding box for the atoms/particles in this system under periodic
        boundary conditions, or a matrix filled with ``0`` for non-periodic systems
        """

        return self._cell
