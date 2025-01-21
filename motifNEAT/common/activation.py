"""Activation functions for NEAT."""
import jax.numpy as jnp
from jax.nn import sigmoid, tanh, relu

class Activation:
    """Collection of activation functions."""
    
    @staticmethod
    def sigmoid(x):
        """Sigmoid activation function."""
        return sigmoid(x)
    
    @staticmethod
    def tanh(x):
        """Tanh activation function."""
        return tanh(x)
    
    @staticmethod
    def relu(x):
        """ReLU activation function."""
        return relu(x)
    
    @staticmethod
    def identity(x):
        """Identity function."""
        return x
    
    @staticmethod
    def step(x):
        """Step function."""
        return jnp.where(x > 0, 1.0, 0.0) 