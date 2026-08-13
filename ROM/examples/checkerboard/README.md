# Checkerboard

Single-group, two-dimensional checkerboard ROM example.

```bash
export PYTHONPATH=/path/to/OpenSnApp/ROM/python
python run_rom_checkerboard.py --exe=/path/to/rom_app_exec --nprocs=4
```

Use `--nprocs`, `--ntrain`, and `--ntest` to size the run. Add `--five-param`
for the source-strength parameter or `--mipod` for optional MIPOD testing.
Runtime files are written to `data/`, `basis/`, `output/`, and `results/`.
