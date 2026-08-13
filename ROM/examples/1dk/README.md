# 1dk

**Location:** [repo root](../../README.md) / [examples/](../README.md) / `1dk/`

1-D k-eigenvalue example and ROM driver. Based off of P58 from [LA-13511](https://doi.org/10.2172/10601)

---

## Files

- `base_P58.py`  
  Base OpenSn deck for problem 58.

- `P58_problem.py`  
  Problem definition and parameterization for problem 58.

- `run_rom_P58.py`  
  Entry-point script that runs the ROM workflow.

- `run_rom_P58_nlke.py`
  Complete active-subspace workflow using the nonlinear k-eigenvalue solver
  for full-order training and validation solves.

- `gradients_P58.py`
  Forward/adjoint gradient deck used to construct the active subspace.

---

## How to Run

From this directory:

```bash
export PYTHONPATH=/path/to/OpenSnApp/ROM/python
# Standard ROM workflow
python run_rom_P58.py --exe=/path/to/rom_app_exec

# Active-subspace ROM workflow
python run_rom_P58.py --exe=/path/to/rom_app_exec --active-subspace

# Full active-subspace workflow with nonlinear k-eigenvalue solves
python run_rom_P58_nlke.py --exe=/path/to/rom_app_exec \
  --num-gradients=10 --ntrain=20 --ntest=2
```


The driver creates `data/`, `basis/`, `output/`, and `results/` in this
directory and stages the static water cross section automatically. Use
`--nprocs`, `--ntrain`, `--ntest`, `--active-rank`, and
`--active-num-gradients` to size the run. After all training points have been
generated, restart at basis merge and system construction with
`--systems-restart`. MIPOD testing is disabled by default; enable it with
`--mipod`.
