# 2gcheckerboard/

**Location:** [repo root](../../README.md) / [examples/](../README.md) / `2gcheckerboard/`

Two-group checkerboard/lattice ROM example.

This case extends the checkerboard configuration to a two-group problem
with separate absorber and scatterer material definitions.

---

## Files

- `base_2gcheckerboard.py`  
  Base two-group OpenSn deck.

- `checkerboard_problem_2g.py`  
  Problem definition and ROM configuration.

- `run_rom_2gcheckerboard.py`  
  Entry-point script for running the ROM workflow.

- `absorber_base.txt`  
  Baseline absorber material definition.

- `scatterer_base.txt`  
  Baseline scatterer material definition.

---

## How to Run

```bash
python run_rom_2gcheckerboard.py --exe=path/to/app/exe
