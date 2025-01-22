import pytest
from motifNEAT.common.config import NEATConfig

@pytest.fixture
def basic_config():
    return NEATConfig(
        num_inputs=2,
        num_outputs=1,
        population_size=10,  # Small population for testing
        num_species=3,
        num_generations=5,
        fitness_threshold=float('inf')
    ) 