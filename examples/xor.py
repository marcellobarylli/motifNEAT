"""Example of training NEAT on the XOR problem."""
import jax.numpy as jnp
from ..motifNEAT.common.config import NEATConfig
from ..motifNEAT.training.experiment import NEATExperiment

def xor_fitness(forward_fn):
    """Fitness function for XOR problem."""
    # XOR inputs and expected outputs
    inputs = jnp.array([
        [0., 0.],
        [0., 1.],
        [1., 0.],
        [1., 1.]
    ])
    expected = jnp.array([0., 1., 1., 0.])
    
    # Get network outputs
    outputs = jnp.array([forward_fn(x)[0] for x in inputs])
    
    # Calculate fitness (negative mean squared error)
    mse = jnp.mean((outputs - expected) ** 2)
    return -mse  # Negative because we want to maximize fitness

def main():
    """Run XOR experiment."""
    # Configure NEAT
    config = NEATConfig(
        num_inputs=2,
        num_outputs=1,
        population_size=150,
        num_species=15,
        num_generations=100,
        fitness_threshold=-0.01  # Stop when MSE < 0.01
    )
    
    # Create and run experiment
    experiment = NEATExperiment(
        config=config,
        fitness_function=xor_fitness,
        wandb_project="neat-xor"  # Optional: track with wandb
    )
    
    best_genome, best_fitness = experiment.train(seed=42)
    
    # Test best genome
    def forward_fn(inputs):
        return experiment.neat.forward(best_genome, inputs)
    
    print("\nTesting best genome:")
    for x1 in [0., 1.]:
        for x2 in [0., 1.]:
            output = forward_fn(jnp.array([x1, x2]))[0]
            print(f"Input: [{x1}, {x2}], Output: {output:.3f}")

if __name__ == "__main__":
    main() 