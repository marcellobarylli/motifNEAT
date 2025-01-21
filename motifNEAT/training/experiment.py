"""Training experiment for NEAT."""
import jax
import jax.numpy as jnp
import numpy as np
from typing import Optional, Callable, Dict, Any
import wandb
from motifNEAT.neat.neat import NEAT
from motifNEAT.common.config import NEATConfig

class NEATExperiment:
    """Experiment class for training NEAT."""
    
    def __init__(self,
                 config: NEATConfig,
                 fitness_function: Callable,
                 wandb_project: Optional[str] = None,
                 wandb_config: Optional[Dict[str, Any]] = None):
        """Initialize experiment."""
        self.config = config
        self.fitness_function = fitness_function
        self.neat = NEAT(config)
        self.wandb_project = wandb_project
        self.wandb_config = wandb_config or {}
    
    def evaluate_population(self, population, state):
        """Evaluate fitness for entire population."""
        fitness = []
        
        for genome in population:
            # Create forward function for this genome
            def forward_fn(inputs):
                return self.neat.forward(genome, inputs)
            
            # Evaluate fitness
            fitness.append(self.fitness_function(forward_fn))
        
        return jnp.array(fitness)
    
    def train(self, seed: int = 0):
        """Run training experiment."""
        # Initialize wandb if specified
        if self.wandb_project:
            wandb.init(
                project=self.wandb_project,
                config=self.wandb_config | {"seed": seed}
            )
        
        # Initialize state
        key = jax.random.PRNGKey(seed)
        state = self.neat.setup(key)
        
        best_fitness = -float('inf')
        best_genome = None
        
        # Training loop
        for generation in range(self.config.num_generations):
            # Get current population
            population = self.neat.ask(state)
            
            # Evaluate population
            fitness = self.evaluate_population(population, state)
            
            # Update best genome if needed
            current_best_idx = jnp.argmax(fitness)
            if fitness[current_best_idx] > best_fitness:
                best_fitness = fitness[current_best_idx]
                best_genome = population[current_best_idx]
            
            # Log metrics
            metrics = {
                'generation': generation,
                'mean_fitness': jnp.mean(fitness),
                'max_fitness': jnp.max(fitness),
                'min_fitness': jnp.min(fitness),
                'std_fitness': jnp.std(fitness),
                'best_fitness_overall': best_fitness,
                'num_species': len(jnp.unique(state.species_members))
            }
            
            print(f"Generation {generation}:")
            for key, value in metrics.items():
                if key != 'generation':
                    print(f"  {key}: {value}")
            
            if self.wandb_project:
                wandb.log(metrics)
            
            # Check if we've reached fitness threshold
            if best_fitness >= self.config.fitness_threshold:
                print(f"\nReached fitness threshold {self.config.fitness_threshold}")
                break
            
            # Create next generation
            state = self.neat.tell(state, fitness)
        
        if self.wandb_project:
            wandb.finish()
        
        return best_genome, best_fitness 