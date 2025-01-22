"""
motifNEAT - A NEAT (NeuroEvolution of Augmenting Topologies) implementation
"""

from .neat import NEAT
from .genome import NodeGene, ConnectionGene
from .training import NEATExperiment
from .common.config import NEATConfig

__version__ = "0.1.0"

__all__ = ["NEAT", "NodeGene", "ConnectionGene", "NEATExperiment", "NEATConfig"]

"""Root package for motifNEAT.""" 