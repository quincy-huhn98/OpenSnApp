# src/rom/

**Location:** [repo root](../../README.md) / [src/](../README.md) / `rom/`

Core reduced-order modeling (ROM) implementation: problem definition, solver(s), and shared ROM data structures.

## Contents

### Problem definition

- [`rom_problem.h`](rom_problem.h), [`rom_problem.cc`](rom_problem.cc)  
  Defines the ROM problem object and its responsibilities (assembly, operators, state, I/O hooks, etc.).

### Solver

- [`steady_state_rom_solver.h`](steady_state_rom_solver.h), [`steady_state_rom_solver.cc`](steady_state_rom_solver.cc)  
  Steady-state ROM solver implementation.

### Shared structs / types

- [`rom_structs.h`](rom_structs.h)  
  Shared ROM data containers and configuration structs.