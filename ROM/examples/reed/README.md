# Reed

Single-group, one-dimensional Reed ROM example.

```bash
export PYTHONPATH=/path/to/OpenSnApp/ROM/python
python run_rom_reed.py --exe=/path/to/rom_app_exec
```

Use `--nprocs`, `--ntrain`, and `--ntest` to size the run. Add `--mipod` to
run optional MIPOD testing. Runtime files are written to `data/`, `basis/`,
`output/`, and `results/` in this directory.
