"""
motifNEAT - A NEAT (NeuroEvolution of Augmenting Topologies) implementation
"""

from .neat import NEAT
from .genome import Gene
from .training import NEATExperiment
from .common.config import NEATConfig

__version__ = "0.1.0"

__all__ = ["NEAT", "Gene", "NEATExperiment", "NEATConfig"]

"""Root package for motifNEAT.""" 