"""Main NEAT algorithm implementation."""
import jax
import jax.numpy as jnp
import equinox as eqx
from typing import List, Tuple, Optional
from ..genome.gene import NodeGene, ConnectionGene
from ...genome.operations.mutation import mutate_genome
from ...genome.operations.crossover import crossover_genome
from . import species as species_ops
from ..common.config import NEATConfig
from ..common.state import State

class NEAT(eqx.Module):
    """NeuroEvolution of Augmenting Topologies (NEAT) implementation."""
    
    config: NEATConfig
    
    def __init__(self, config: NEATConfig):
        """Initialize NEAT algorithm."""
        self.config = config
    
    def setup(self, key: jnp.ndarray) -> State:
        """Initialize the population and state."""
        state = State(key)
        
        # Create initial population
        population = self._create_initial_population(state.randkey)
        
        # Initialize species
        species, state = species_ops.speciate(population, self.config, state)
        
        return state
    
    def _create_initial_population(self, key: jnp.ndarray) -> List[Tuple[List[NodeGene], List[ConnectionGene]]]:
        """Create initial population with minimal networks."""
        keys = jax.random.split(key, self.config.population_size)
        population = []
        
        for k in keys:
            # Create input and output nodes
            nodes = []
            for i in range(self.config.num_inputs):
                nodes.append(NodeGene(i, 'input', layer=0))
            
            for i in range(self.config.num_outputs):
                nodes.append(NodeGene(
                    self.config.num_inputs + i,
                    'output',
                    activation=self.config.activation_default,
                    layer=1
                ))
            
            # Create initial connections (fully connected)
            connections = []
            conn_key, weight_key = jax.random.split(k)
            weights = jax.random.normal(weight_key, (self.config.num_inputs, self.config.num_outputs))
            
            innovation = 0
            for i in range(self.config.num_inputs):
                for j in range(self.config.num_outputs):
                    connections.append(ConnectionGene(
                        i,
                        self.config.num_inputs + j,
                        weight=weights[i, j],
                        innovation=innovation
                    ))
                    innovation += 1
            
            population.append((nodes, connections))
        
        return population
    
    def ask(self, state: State) -> List[Tuple[List[NodeGene], List[ConnectionGene]]]:
        """Get current population for evaluation."""
        return list(zip(state.population_nodes, state.population_connections))
    
    def tell(self, state: State, fitness: jnp.ndarray) -> State:
        """Update population based on fitness results."""
        # Update state with fitness
        state = state.update(
            generation=state.generation + 1,
            best_fitness=jnp.maximum(state.best_fitness, jnp.max(fitness))
        )
        
        # Calculate species fitness
        species_fitness = species_ops.get_species_fitness(
            fitness, state.species_members
        )
        
        # Calculate number of offspring per species
        offspring_counts = species_ops.get_offspring_counts(
            species_fitness, self.config
        )
        
        # Create next generation
        next_population = self._create_next_generation(
            state, fitness, offspring_counts
        )
        
        # Speciate new population
        species, state = species_ops.speciate(
            next_population, self.config, state
        )
        
        return state
    
    def _create_next_generation(self,
                              state: State,
                              fitness: jnp.ndarray,
                              offspring_counts: jnp.ndarray) -> List[Tuple[List[NodeGene], List[ConnectionGene]]]:
        """Create next generation through selection, crossover, and mutation."""
        next_population = []
        key = state.randkey
        
        # For each species
        for species_idx, offspring_count in enumerate(offspring_counts):
            # Get members of this species
            species_mask = state.species_members == species_idx
            species_members = jnp.where(species_mask)[0]
            species_fitness = fitness[species_mask]
            
            # Sort by fitness
            sorted_idx = jnp.argsort(species_fitness)[::-1]
            species_members = species_members[sorted_idx]
            
            # Keep elite members
            elite_count = min(self.config.species_elitism, len(species_members))
            for i in range(elite_count):
                member_idx = species_members[i]
                next_population.append((
                    state.population_nodes[member_idx],
                    state.population_connections[member_idx]
                ))
            
            # Create offspring for remaining slots
            for _ in range(offspring_count - elite_count):
                # Select parents
                k1, k2, k3 = jax.random.split(key, 3)
                parent1_idx = species_members[jax.random.randint(k1, (), 0, len(species_members))]
                parent2_idx = species_members[jax.random.randint(k2, (), 0, len(species_members))]
                
                # Perform crossover
                child_nodes, child_conns = crossover_genome(
                    state.population_nodes[parent1_idx],
                    state.population_connections[parent1_idx],
                    state.population_nodes[parent2_idx],
                    state.population_connections[parent2_idx],
                    fitness[parent1_idx],
                    fitness[parent2_idx],
                    self.config,
                    k3
                )
                
                # Perform mutation
                child_nodes, child_conns = mutate_genome(
                    child_nodes, child_conns,
                    self.config, k3
                )
                
                next_population.append((child_nodes, child_conns))
        
        return next_population
    
    def forward(self, genome: Tuple[List[NodeGene], List[ConnectionGene]], inputs: jnp.ndarray) -> jnp.ndarray:
        """Forward pass through the network."""
        nodes, connections = genome
        
        # Initialize node values
        node_values = jnp.zeros(len(nodes))
        node_values = node_values.at[:self.config.num_inputs].set(inputs)
        
        # Process nodes by layer
        max_layer = max(node.layer for node in nodes)
        for layer in range(1, max_layer + 1):
            layer_nodes = [i for i, node in enumerate(nodes) if node.layer == layer]
            
            # For each node in this layer
            for node_idx in layer_nodes:
                # Get incoming connections
                incoming = [conn for conn in connections 
                          if conn.output_node == node_idx and conn.enabled]
                
                if not incoming:
                    continue
                
                # Sum inputs * weights
                node_inputs = jnp.array([node_values[conn.input_node] * conn.weight 
                                       for conn in incoming])
                
                # Apply activation
                node = nodes[node_idx]
                node_values = node_values.at[node_idx].set(
                    node.forward(self.config.aggregation_function(node_inputs))
                )
        
        # Return output values
        return node_values[self.config.num_inputs:self.config.num_inputs + self.config.num_outputs] 