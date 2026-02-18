# python/

**Location:** [repo root](../README.md) / `python/`

Python drivers and utilities for running example problems and ROM workflows using the application implemented in [`src/`](../src/README.md).

This folder is intended to:
- standardize launching (local vs batch/HPC),
- keep ROM pipeline orchestration in one place,
- centralize plotting and small utilities.

## Key modules

- [`rom_driver.py`](rom_driver.py)  
  Orchestrates the ROM pipeline (e.g., offline, merge, system, online phases).

- [`job_manager.py`](job_manager.py)  
  Launch abstraction for running the executable (local execution vs MPI/Slurm-style workflows).

- [`plotting.py`](plotting.py)  
  Plotting helpers for example runs and ROM result summaries.

- [`utils.py`](utils.py)  
  Shared utilities (loading fluxes from h5, basic sampling capabilities, and updating OpenSn .xs files).