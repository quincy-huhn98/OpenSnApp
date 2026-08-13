# C5G7

Two-dimensional C5G7 k-eigenvalue ROM example with optional active-subspace
sampling. The active-subspace workflow computes forward/adjoint gradients for
the four fuel materials before constructing the reduced parameter space.

## How to run

From this directory:

```bash
export PYTHONPATH=/path/to/OpenSnApp/ROM/python
# Standard ROM workflow
python run_rom_c5g7.py --exe=/path/to/rom_app_exec --nprocs=48

# Active-subspace ROM workflow
python run_rom_c5g7.py --exe=/path/to/rom_app_exec --active-subspace \
  --active-rank=3 --nprocs=48
```

The driver creates `data/`, `basis/`, `output/`, and `results/` here and
automatically stages the unparameterized material files. Use `--active-rank`
to select the active dimension. After all training points have been generated,
restart at basis merge and system construction with `--systems-restart`.
MIPOD testing is disabled by default; enable it with `--mipod`.