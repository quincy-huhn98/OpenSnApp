# Two-group checkerboard

Two-group checkerboard ROM example using parameterized material files.

```bash
export PYTHONPATH=/path/to/OpenSnApp/ROM/python
python run_rom_2gcheckerboard.py --exe=/path/to/rom_app_exec --nprocs=4
```

Use `--nprocs`, `--ntrain`, and `--ntest` to size the run. Add `--mipod` to
run optional MIPOD testing. Runtime files are written to `data/`, `basis/`,
`output/`, and `results/`.
