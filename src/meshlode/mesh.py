from typing import Optional

import torch
from metatensor.torch import TensorBlock

from .system import System


class FieldBuilder(torch.nn.Module):
    def __init__(self):
        pass

    def compute(self, system: System, embeddings: Optional[TensorBlock] = None):
        pass


class MeshInterpolate(torch.nn.Module):
    pass


class FieldProjector(torch.nn.Module):
    pass
