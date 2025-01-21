# NEAT Standalone Implementation

A clean, standalone implementation of NeuroEvolution of Augmenting Topologies (NEAT) using JAX and Equinox.

## Installation

```bash
pip install -r requirements.txt
```

## Project Structure

```
motifNEAT/
├── neat/           # Core NEAT algorithm implementation
├── genome/         # Genome, node, and connection implementations
├── common/         # Common utilities and functions
└── training/       # Training infrastructure
```

## Usage

Basic example of training NEAT on XOR:

```python
from motifNEAT.training.experiment import NEATExperiment
from motifNEAT.common.config import NEATConfig

# Configure NEAT
config = NEATConfig(
    num_inputs=2,
    num_outputs=1,
    population_size=150,
    num_species=15
)

# Create and run experiment
experiment = NEATExperiment(config)
experiment.train()
```

## Features

- Full NEAT implementation with speciation
- JAX-based for efficient computation
- Clean, modular architecture
- Built-in visualization tools
- Checkpointing and experiment tracking 