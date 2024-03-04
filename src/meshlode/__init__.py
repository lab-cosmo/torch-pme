from .calculators.meshpotential import MeshPotential

try:
    from . import metatensor  # noqa
except ImportError:
    pass


__all__ = ["MeshPotential"]
__version__ = "0.0.0-dev"
