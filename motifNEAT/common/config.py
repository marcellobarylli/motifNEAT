"""Configuration for NEAT."""
from dataclasses import dataclass
from typing import Tuple, Callable
from .activation import Activation

@dataclass
class NEATConfig:
    """Configuration for NEAT algorithm."""
    
    # Network structure
    num_inputs: int
    num_outputs: int
    max_nodes: int = 100
    
    # Population
    population_size: int = 150
    num_species: int = 15
    
    # Species
    compatibility_threshold: float = 3.0
    compatibility_disjoint_coefficient: float = 1.0
    compatibility_weight_coefficient: float = 0.4
    species_elitism: int = 2
    species_survival_threshold: float = 0.2
    
    # Genetics
    weight_mutation_rate: float = 0.8
    weight_perturbation_power: float = 0.5
    node_add_prob: float = 0.03
    connection_add_prob: float = 0.05
    node_delete_prob: float = 0.01
    connection_delete_prob: float = 0.02
    
    # Node genes
    activation_options: Tuple[Callable, ...] = (
        Activation.sigmoid,
        Activation.tanh,
        Activation.relu
    )
    activation_default: Callable = Activation.sigmoid
    aggregation_function: Callable = lambda x: x.sum(axis=0)
    
    # Training
    num_generations: int = 100
    fitness_threshold: float = float('inf')  # No threshold by default
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        assert self.num_inputs > 0, "Number of inputs must be positive"
        assert self.num_outputs > 0, "Number of outputs must be positive"
        assert self.population_size > 0, "Population size must be positive"
        assert self.num_species > 0, "Number of species must be positive"
        assert 0 <= self.species_survival_threshold <= 1, "Species survival threshold must be between 0 and 1"
        assert self.compatibility_threshold > 0, "Compatibility threshold must be positive" 