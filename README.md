# sbo_mmre: Simulation Framework for ISRS and Rover SBO Environments

This repository provides simulation tools for two main environments:
- **Rover SBO Environment** (in `sbo_rover`)
- **ISRS Environment** (in `sbo_isrs`)

Each environment supports running and comparing multiple decision-making policies, including advanced POMDP solvers and baselines.

---

## Table of Contents
- [Project Structure](#project-structure)
- [Environment Setup](#environment-setup)
- [Running Simulations](#running-simulations)
  - [Rover SBO Environment](#rover-sbo-environment)
  - [ISRS Environment](#isrs-environment)
- [Policies and Algorithms](#policies-and-algorithms)
  - [Rover SBO Policies](#rover-sbo-policies)
  - [ISRS Policies](#isrs-policies)
- [Testing](#testing)
- [Code Style](#code-style)
- [Notes](#notes)

---

## Project Structure

```
sbo_mmre/
  sbo_rover/         # Rover SBO simulation code and scripts
    scripts/
      run_simulations.py
    src/
      rover_sbo/
        ...
    requirements.txt
    pyproject.toml
  sbo_isrs/          # ISRS simulation code and scripts
    scripts/
      run_simulations.py
    src/
      ...
    pyproject.toml
```

---

## Environment Setup

We recommend using **conda** for environment management.

### 1. Create and Activate Conda Environment

```bash
conda create -n sbo_env python=3.9
conda activate sbo_env
```

### 2. Install Requirements

#### For Rover SBO Environment
```bash
cd sbo_mmre/sbo_rover
pip install -r requirements.txt
```

#### For ISRS Environment
```bash
cd sbo_mmre/sbo_isrs
pip install .
```
Or, to install dependencies only:
```bash
pip install -r <(grep -oP '"\\K[^"]+' pyproject.toml | grep -vE '^[0-9.]+$')
```

---

## Running Simulations

### Rover SBO Environment

- **Script:** `sbo_mmre/sbo_rover/scripts/run_simulations.py`
- **Usage Example:**

```bash
cd sbo_mmre/sbo_rover/scripts
python run_simulations.py --grid_size 10 10 --num_sample_types 3 --num_episodes 10 --max_steps 100 --seed 42 --policy_types pomcp basic enhanced raster
```

- **Arguments:**
  - `--grid_size`: Grid size (rows cols)
  - `--num_sample_types`: Number of sample types
  - `--num_episodes`: Number of episodes
  - `--max_steps`: Maximum steps per episode
  - `--seed`: Random seed
  - `--policy_types`: List of policies to evaluate (see below)

### ISRS Environment

- **Script:** `sbo_mmre/sbo_isrs/scripts/run_simulations.py`
- **Usage Example:**

```bash
cd sbo_mmre/sbo_isrs/scripts
python run_simulations.py --num_locations 10 --num_good 3 --num_bad 3 --num_beacons 2 --sensor_efficiency 0.8 --num_episodes 10 --max_steps 100 --seed 42 --policy_types random greedy pomcp information dpw
```

- **Arguments:**
  - `--num_locations`: Number of locations
  - `--num_good`: Number of good samples
  - `--num_bad`: Number of bad samples
  - `--num_beacons`: Number of beacons
  - `--sensor_efficiency`: Sensor efficiency
  - `--num_episodes`: Number of episodes
  - `--max_steps`: Maximum steps per episode
  - `--seed`: Random seed
  - `--policy_types`: List of policies to evaluate (see below)

---

## Policies and Algorithms

### Rover SBO Policies

- **POMCP**: Partially Observable Monte Carlo Planning. Uses Monte Carlo Tree Search (MCTS) for planning under uncertainty.
- **Basic**: Greedy policy with random exploration. Selects actions with highest expected value, occasionally explores randomly.
- **Enhanced**: Enhanced GP-MCTS (Gaussian Process Monte Carlo Tree Search). Uses Gaussian Processes for spatial modeling and adaptive MCTS for efficient exploration and exploitation.
- **Raster**: Follows a systematic raster scan pattern across the grid, periodically drilling.

### ISRS Policies

- **random**: Selects actions uniformly at random.
- **greedy**: Chooses actions that maximize immediate reward based on the current belief.
- **pomcp**: Partially Observable Monte Carlo Planning. Uses MCTS for planning in the ISRS POMDP.
- **information**: Information-seeking policy. Selects actions that maximize expected information gain (e.g., reduction in belief uncertainty).
- **dpw**: POMCP with Double Progressive Widening (DPW). Enhances POMCP by controlling the branching factor in large or continuous action/observation spaces.


---

## Notes
- For advanced usage, see the docstrings in each script and policy class.
- For reproducibility, always set the `--seed` argument.
- A few policies are implemented in the codebase but are not used in the main simulation scripts. You can explore or extend these policies as needed.
- For questions or issues, please open an issue or consult the code documentation. 
