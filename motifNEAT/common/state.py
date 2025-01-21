"""State management for NEAT."""
import jax.numpy as jnp
import equinox as eqx

class State(eqx.Module):
    """State class for managing NEAT algorithm state."""
    
    randkey: jnp.ndarray
    generation: int
    best_fitness: float
    population_nodes: jnp.ndarray
    population_connections: jnp.ndarray
    species_keys: jnp.ndarray
    species_fitness: jnp.ndarray
    species_members: jnp.ndarray
    
    def __init__(self, randkey):
        """Initialize state with a random key."""
        self.randkey = randkey
        self.generation = 0
        self.best_fitness = -jnp.inf
        self.population_nodes = jnp.array([])
        self.population_connections = jnp.array([])
        self.species_keys = jnp.array([])
        self.species_fitness = jnp.array([])
        self.species_members = jnp.array([])
    
    def update(self, **kwargs):
        """Update state with new values."""
        return eqx.tree_at(lambda s: [getattr(s, k) for k in kwargs.keys()], 
                          self, 
                          [v for v in kwargs.values()])
    
    @property
    def state_dict(self):
        """Get state as dictionary."""
        return {
            'generation': self.generation,
            'best_fitness': self.best_fitness,
            'population_nodes': self.population_nodes,
            'population_connections': self.population_connections,
            'species_keys': self.species_keys,
            'species_fitness': self.species_fitness,
            'species_members': self.species_members
        } 