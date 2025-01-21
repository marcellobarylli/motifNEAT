"""Species management for NEAT."""
import jax
import jax.numpy as jnp
from typing import List, Tuple
from ..genome.gene import NodeGene, ConnectionGene
from ..common.config import NEATConfig
from ..common.state import State

def calculate_genome_distance(genome1: Tuple[List[NodeGene], List[ConnectionGene]],
                            genome2: Tuple[List[NodeGene], List[ConnectionGene]],
                            config: NEATConfig) -> float:
    """Calculate the compatibility distance between two genomes."""
    nodes1, conns1 = genome1
    nodes2, conns2 = genome2
    
    # Count matching and disjoint genes
    node_matches = sum(1 for n1 in nodes1 if any(n2.node_id == n1.node_id for n2 in nodes2))
    conn_matches = sum(1 for c1 in conns1 if any(c2.innovation == c1.innovation for c2 in conns2))
    
    node_disjoint = len(nodes1) + len(nodes2) - 2 * node_matches
    conn_disjoint = len(conns1) + len(conns2) - 2 * conn_matches
    
    # Calculate average weight difference for matching connections
    weight_diff = 0.0
    num_matching = 0
    for c1 in conns1:
        for c2 in conns2:
            if c1.innovation == c2.innovation:
                weight_diff += abs(c1.weight - c2.weight)
                num_matching += 1
    
    avg_weight_diff = weight_diff / max(1, num_matching)
    
    # Calculate compatibility distance
    distance = (config.compatibility_disjoint_coefficient * (node_disjoint + conn_disjoint) +
               config.compatibility_weight_coefficient * avg_weight_diff)
    
    return distance

def speciate(population: List[Tuple[List[NodeGene], List[ConnectionGene]]],
             config: NEATConfig,
             state: State) -> Tuple[List[List[int]], State]:
    """Divide population into species."""
    species = []
    species_representatives = []
    
    # Get random key
    key = state.randkey
    
    # For each genome in the population
    for i, genome in enumerate(population):
        # Find the first species this genome belongs to
        found_species = False
        for j, representative in enumerate(species_representatives):
            distance = calculate_genome_distance(genome, representative, config)
            if distance < config.compatibility_threshold:
                species[j].append(i)
                found_species = True
                break
        
        # If no compatible species found, create a new one
        if not found_species:
            species.append([i])
            species_representatives.append(genome)
    
    # Update species information in state
    species_keys = jnp.arange(len(species))
    species_members = jnp.zeros((config.population_size,), dtype=jnp.int32)
    
    for i, specie in enumerate(species):
        for member in specie:
            species_members = species_members.at[member].set(i)
    
    state = state.update(
        species_keys=species_keys,
        species_members=species_members
    )
    
    return species, state

def compute_adjusted_fitness(raw_fitness: jnp.ndarray,
                           species_members: jnp.ndarray) -> jnp.ndarray:
    """Compute adjusted fitness using explicit fitness sharing."""
    # Count number of members in each species
    unique_species = jnp.unique(species_members)
    species_sizes = jnp.zeros_like(raw_fitness)
    
    for s in unique_species:
        size = jnp.sum(species_members == s)
        species_sizes = jnp.where(species_members == s, size, species_sizes)
    
    # Compute adjusted fitness
    adjusted_fitness = raw_fitness / species_sizes
    return adjusted_fitness

def get_species_fitness(raw_fitness: jnp.ndarray,
                       species_members: jnp.ndarray) -> jnp.ndarray:
    """Get average adjusted fitness for each species."""
    adjusted_fitness = compute_adjusted_fitness(raw_fitness, species_members)
    
    unique_species = jnp.unique(species_members)
    species_fitness = jnp.zeros(len(unique_species))
    
    for i, s in enumerate(unique_species):
        species_mask = species_members == s
        species_fitness = species_fitness.at[i].set(
            jnp.mean(adjusted_fitness[species_mask])
        )
    
    return species_fitness

def get_offspring_counts(species_fitness: jnp.ndarray,
                        config: NEATConfig) -> jnp.ndarray:
    """Compute number of offspring for each species."""
    # Normalize fitness
    total_fitness = jnp.sum(species_fitness)
    normalized_fitness = species_fitness / total_fitness
    
    # Calculate offspring counts
    offspring_counts = jnp.floor(normalized_fitness * config.population_size)
    
    # Ensure at least one member survives if species is not stagnant
    offspring_counts = jnp.maximum(offspring_counts, 1)
    
    # Adjust to match population size
    total_offspring = jnp.sum(offspring_counts)
    if total_offspring < config.population_size:
        # Add remaining to fittest species
        difference = config.population_size - total_offspring
        max_fitness_idx = jnp.argmax(species_fitness)
        offspring_counts = offspring_counts.at[max_fitness_idx].add(difference)
    
    return offspring_counts.astype(jnp.int32) 