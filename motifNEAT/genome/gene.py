"""Gene implementations for NEAT."""
import jax.numpy as jnp
import equinox as eqx
from typing import Callable, Optional
from ..common.activation import Activation

class NodeGene(eqx.Module):
    """Node gene implementation."""
    
    node_id: int
    node_type: str  # 'input', 'hidden', or 'output'
    activation: Callable
    bias: float
    response: float
    layer: int
    
    def __init__(self, 
                 node_id: int,
                 node_type: str,
                 activation: Optional[Callable] = None,
                 bias: float = 0.0,
                 response: float = 1.0,
                 layer: int = 0):
        """Initialize node gene."""
        self.node_id = node_id
        self.node_type = node_type
        self.activation = activation or Activation.sigmoid
        self.bias = bias
        self.response = response
        self.layer = layer
    
    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass through the node."""
        return self.activation(self.response * x + self.bias)
    
    def distance(self, other: 'NodeGene') -> float:
        """Calculate distance between two node genes."""
        if self.node_id != other.node_id:
            return float('inf')
        return abs(self.bias - other.bias) + abs(self.response - other.response)

class ConnectionGene(eqx.Module):
    """Connection gene implementation."""
    
    input_node: int
    output_node: int
    weight: float
    enabled: bool
    innovation: int
    
    def __init__(self,
                 input_node: int,
                 output_node: int,
                 weight: float = 0.0,
                 enabled: bool = True,
                 innovation: int = 0):
        """Initialize connection gene."""
        self.input_node = input_node
        self.output_node = output_node
        self.weight = weight
        self.enabled = enabled
        self.innovation = innovation
    
    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass through the connection."""
        return jnp.where(self.enabled, x * self.weight, 0.0)
    
    def distance(self, other: 'ConnectionGene') -> float:
        """Calculate distance between two connection genes."""
        if self.innovation != other.innovation:
            return float('inf')
        return abs(self.weight - other.weight) 