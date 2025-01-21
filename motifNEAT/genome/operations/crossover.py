"""Crossover operations for NEAT."""
import jax
import jax.numpy as jnp
from typing import Tuple, List, Dict
from ..gene import NodeGene, ConnectionGene
from ...motifNEAT.common.config import NEATConfig

def align_genes(parent1_genes: Dict[int, NodeGene],
               parent2_genes: Dict[int, NodeGene]) -> Tuple[List[Tuple[NodeGene, NodeGene]], 
                                                          List[NodeGene], 
                                                          List[NodeGene]]:
    """Align genes from two parents."""
    matching = []
    disjoint1 = []
    disjoint2 = []
    
    all_keys = set(parent1_genes.keys()) | set(parent2_genes.keys())
    
    for key in sorted(all_keys):
        gene1 = parent1_genes.get(key)
        gene2 = parent2_genes.get(key)
        
        if gene1 and gene2:
            matching.append((gene1, gene2))
        elif gene1:
            disjoint1.append(gene1)
        else:
            disjoint2.append(gene2)
    
    return matching, disjoint1, disjoint2

def crossover_nodes(parent1_nodes: List[NodeGene],
                   parent2_nodes: List[NodeGene],
                   parent1_fitness: float,
                   parent2_fitness: float,
                   key: jnp.ndarray) -> List[NodeGene]:
    """Perform crossover of node genes."""
    # Convert to dictionaries for easier matching
    p1_dict = {node.node_id: node for node in parent1_nodes}
    p2_dict = {node.node_id: node for node in parent2_nodes}
    
    matching, disjoint1, disjoint2 = align_genes(p1_dict, p2_dict)
    
    # Initialize child nodes with matching genes
    child_nodes = []
    for n1, n2 in matching:
        # Randomly choose between matching genes
        if jax.random.uniform(key) < 0.5:
            child_nodes.append(n1)
        else:
            child_nodes.append(n2)
    
    # Add disjoint genes from the fitter parent
    if parent1_fitness > parent2_fitness:
        child_nodes.extend(disjoint1)
    else:
        child_nodes.extend(disjoint2)
    
    # Sort by node_id to maintain consistent ordering
    return sorted(child_nodes, key=lambda x: x.node_id)

def crossover_connections(parent1_conns: List[ConnectionGene],
                        parent2_conns: List[ConnectionGene],
                        parent1_fitness: float,
                        parent2_fitness: float,
                        key: jnp.ndarray) -> List[ConnectionGene]:
    """Perform crossover of connection genes."""
    # Convert to dictionaries for easier matching
    p1_dict = {conn.innovation: conn for conn in parent1_conns}
    p2_dict = {conn.innovation: conn for conn in parent2_conns}
    
    matching, disjoint1, disjoint2 = align_genes(p1_dict, p2_dict)
    
    # Initialize child connections with matching genes
    child_conns = []
    for c1, c2 in matching:
        # Randomly choose between matching genes
        if jax.random.uniform(key) < 0.5:
            child_conns.append(c1)
        else:
            child_conns.append(c2)
    
    # Add disjoint genes from the fitter parent
    if parent1_fitness > parent2_fitness:
        child_conns.extend(disjoint1)
    else:
        child_conns.extend(disjoint2)
    
    # Sort by innovation number to maintain consistent ordering
    return sorted(child_conns, key=lambda x: x.innovation)

def crossover_genome(parent1_nodes: List[NodeGene],
                    parent1_conns: List[ConnectionGene],
                    parent2_nodes: List[NodeGene],
                    parent2_conns: List[ConnectionGene],
                    parent1_fitness: float,
                    parent2_fitness: float,
                    config: NEATConfig,
                    key: jnp.ndarray) -> Tuple[List[NodeGene], List[ConnectionGene]]:
    """Perform crossover between two parent genomes."""
    k1, k2 = jax.random.split(key)
    
    child_nodes = crossover_nodes(
        parent1_nodes, parent2_nodes,
        parent1_fitness, parent2_fitness,
        k1
    )
    
    child_conns = crossover_connections(
        parent1_conns, parent2_conns,
        parent1_fitness, parent2_fitness,
        k2
    )
    
    return child_nodes, child_conns 