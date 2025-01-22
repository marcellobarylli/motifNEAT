"""Mutation operations for NEAT."""
import jax
import jax.numpy as jnp
from typing import Tuple, List
from ..gene import NodeGene, ConnectionGene
from ...common.config import NEATConfig

def mutate_weight(connection: ConnectionGene, 
                 config: NEATConfig, 
                 key: jnp.ndarray) -> ConnectionGene:
    """Mutate connection weight."""
    k1, k2 = jax.random.split(key)
    
    # Decide whether to mutate
    should_mutate = jax.random.uniform(k1) < config.weight_mutation_rate
    
    # Generate perturbation
    perturbation = jax.random.normal(k2) * config.weight_perturbation_power
    
    # Apply mutation
    new_weight = jnp.where(should_mutate,
                          connection.weight + perturbation,
                          connection.weight)
    
    return connection.replace(weight=new_weight)

def add_node(nodes: List[NodeGene],
            connections: List[ConnectionGene],
            config: NEATConfig,
            key: jnp.ndarray) -> Tuple[List[NodeGene], List[ConnectionGene]]:
    """Add a new node by splitting an existing connection."""
    if not connections:
        return nodes, connections
    
    k1, k2, k3 = jax.random.split(key, 3)
    
    # Select random connection to split
    conn_idx = jax.random.randint(k1, (), 0, len(connections))
    conn = connections[conn_idx]
    
    if not conn.enabled:
        return nodes, connections
    
    # Create new node
    new_node_id = len(nodes)
    new_node = NodeGene(
        node_id=new_node_id,
        node_type='hidden',
        activation=jax.random.choice(k2, config.activation_options),
        layer=(nodes[conn.input_node].layer + nodes[conn.output_node].layer) // 2
    )
    
    # Create new connections
    new_conn1 = ConnectionGene(
        input_node=conn.input_node,
        output_node=new_node_id,
        weight=1.0,
        innovation=len(connections)
    )
    
    new_conn2 = ConnectionGene(
        input_node=new_node_id,
        output_node=conn.output_node,
        weight=conn.weight,
        innovation=len(connections) + 1
    )
    
    # Disable old connection
    connections[conn_idx] = conn.replace(enabled=False)
    
    return nodes + [new_node], connections + [new_conn1, new_conn2]

def add_connection(nodes: List[NodeGene],
                  connections: List[ConnectionGene],
                  config: NEATConfig,
                  key: jnp.ndarray) -> List[ConnectionGene]:
    """Add a new connection between existing nodes."""
    if len(nodes) < 2:
        return connections
    
    k1, k2 = jax.random.split(key)
    
    # Select random nodes
    input_idx = jax.random.randint(k1, (), 0, len(nodes))
    output_idx = jax.random.randint(k2, (), 0, len(nodes))
    
    # Check if connection already exists or would create cycle
    for conn in connections:
        if (conn.input_node == input_idx and 
            conn.output_node == output_idx):
            return connections
    
    if nodes[input_idx].layer >= nodes[output_idx].layer:
        return connections
    
    # Create new connection
    new_conn = ConnectionGene(
        input_node=input_idx,
        output_node=output_idx,
        weight=jax.random.normal(key),
        innovation=len(connections)
    )
    
    return connections + [new_conn]

def mutate_genome(nodes: List[NodeGene],
                 connections: List[ConnectionGene],
                 config: NEATConfig,
                 key: jnp.ndarray) -> Tuple[List[NodeGene], List[ConnectionGene]]:
    """Apply mutation operations to genome."""
    keys = jax.random.split(key, 4)
    
    # Mutate weights
    connections = [mutate_weight(conn, config, keys[0]) 
                  for conn in connections]
    
    # Add node
    if jax.random.uniform(keys[1]) < config.node_add_prob:
        nodes, connections = add_node(nodes, connections, config, keys[2])
    
    # Add connection
    if jax.random.uniform(keys[3]) < config.connection_add_prob:
        connections = add_connection(nodes, connections, config, keys[3])
    
    return nodes, connections 