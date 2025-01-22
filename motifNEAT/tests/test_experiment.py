import pytest
import jax.numpy as jnp
from motifNEAT.common.config import NEATConfig
from motifNEAT.training.experiment import NEATExperiment
from motifNEAT.tests.conftest import basic_config

def simple_fitness_function(forward_fn):
    """Simple fitness function that rewards outputs close to 0.5."""
    inputs = jnp.array([[0.0, 0.0], [1.0, 1.0]])  # Two inputs per sample
    outputs = jnp.array([forward_fn(x)[0] for x in inputs])
    return -jnp.mean((outputs - 0.5) ** 2)  # Negative MSE from target 0.5

class TestNEATExperiment:
    def test_experiment_initialization(self, basic_config):
        """Test that experiment initializes correctly."""
        experiment = NEATExperiment(
            config=basic_config,
            fitness_function=simple_fitness_function
        )
        assert experiment.config == basic_config
        assert experiment.fitness_function == simple_fitness_function
        assert experiment.wandb_project is None

    def test_training_run(self, basic_config):
        """Test that training runs without errors."""
        experiment = NEATExperiment(
            config=basic_config,
            fitness_function=simple_fitness_function
        )
        
        best_genome, best_fitness = experiment.train(seed=42)
        
        assert best_genome is not None
        assert isinstance(best_fitness, float)
        assert best_fitness > -float('inf')

    def test_config_validation(self):
        """Test that invalid configs raise appropriate errors."""
        with pytest.raises(AssertionError):
            NEATConfig(
                num_inputs=-1,  # Invalid: must be positive
                num_outputs=1,
                population_size=10
            )
        
        with pytest.raises(AssertionError):
            NEATConfig(
                num_inputs=1,
                num_outputs=1,
                population_size=10,
                species_survival_threshold=1.5  # Invalid: must be between 0 and 1
            )

    def test_reproducibility(self, basic_config):
        """Test that experiments with same seed produce same results."""
        experiment1 = NEATExperiment(
            config=basic_config,
            fitness_function=simple_fitness_function
        )
        experiment2 = NEATExperiment(
            config=basic_config,
            fitness_function=simple_fitness_function
        )
        
        _, fitness1 = experiment1.train(seed=42)
        _, fitness2 = experiment2.train(seed=42)
        
        assert fitness1 == fitness2

    def test_different_seeds(self, basic_config):
        """Test that different seeds produce different results."""
        experiment = NEATExperiment(
            config=basic_config,
            fitness_function=simple_fitness_function
        )
        
        _, fitness1 = experiment.train(seed=42)
        _, fitness2 = experiment.train(seed=43)
        
        assert fitness1 != fitness2 

def main():
    # Create test instance
    test_instance = TestNEATExperiment()
    
    # Get basic_config fixture
    config = NEATConfig(
        num_inputs=2,
        num_outputs=1,
        population_size=10,  # Small population for testing
        num_species=3,
        num_generations=5,
        fitness_threshold=float('inf')
    )
    
    print("Running experiment initialization test...")
    # breakpoint()  # Debug point 1
    test_instance.test_experiment_initialization(config)
    print("Initialization test passed!")
    
    print("\nRunning training test...")
    # breakpoint()  # Debug point 2
    test_instance.test_training_run(config)
    print("Training test passed!")
    
    print("\nRunning config validation test...")
    # breakpoint()  # Debug point 3
    test_instance.test_config_validation()
    print("Config validation test passed!")

if __name__ == "__main__":
    main()