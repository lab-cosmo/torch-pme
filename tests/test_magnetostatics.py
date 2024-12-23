import torch

from torchpme.calculators import CalculatorDipole
from torchpme.potentials import PotentialDipole


def test_magnetostatics():
    calculator = CalculatorDipole(
        potential=PotentialDipole(),
        full_neighbor_list=False,
    )
    dipoles = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    pot = calculator(
        dipoles=dipoles,
        cell=torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]),
        positions=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 4.0]]),
        neighbor_indices=torch.tensor([[1, 0], [1, 2], [0, 2]]),
        neighbor_vectors=torch.tensor(
            [[0.0, 2.0, 0.0], [0.0, 2.0, 0.0], [0.0, 4.0, 0.0]]
        ),
    )
    result = torch.einsum("ij,ij->i", pot, dipoles).sum()
    expected_result = torch.tensor(-0.2656)
    assert torch.isclose(
        result, expected_result, atol=1e-4
    ), f"Expected {expected_result}, but got {result}"
