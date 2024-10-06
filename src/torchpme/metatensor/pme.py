from .. import calculators as torch_calculators
from .base import Calculator


class PMECalculator(Calculator):
    r"""
    Potential using a particle mesh-based Ewald (PME).

    Refer to :py:class:`torchpme.PMECalculator` for parameter documentation.

    Example
    -------
    We calculate the Madelung constant of a CsCl (Cesium-Chloride) crystal. The
    reference value is :math:`2 \cdot 1.7626 / \sqrt{3} \approx 2.0354`.

    >>> from torchpme.metatensor import get_cscl_data, PMECalculator
    >>> from torchpme import CoulombPotential

    Define a simple CsCl example structure

    >>> system, neighbors = get_cscl_data()

    If you inspect the neighbor list you will notice that the TensorBlock is empty for
    the given system, which means the the whole potential will be calculated using the
    long range part of the potential.

    >>> pot = CoulombPotential(range_radius=0.1)

    Finally, we initlize the potential class and
    ``compute`` the potential for the crystal.

    >>> pme = PMECalculator(potential=pot)
    >>> potential = pme.forward(system=system, neighbors=neighbors)

    The results are stored inside the ``values`` property inside the first
    :py:class:`TensorBlock <metatensor.torch.TensorBlock>` of the ``potential``.

    >>> potential[0].values
    tensor([[-1.0192],
            [ 1.0192]], dtype=torch.float64)

    Which is close to the reference value given above.

    """

    # see torchpme.metatensor.base
    _base_calculator = torch_calculators.PMECalculator
