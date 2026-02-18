Standalone ROM application built around OpenSn and libROM, with supporting Python tooling and example input decks.

This repo is organized to:
- build a C++ application/binary that links against OpenSn and libROM,
- run transport problems and ROM workflows from Python drivers,
- keep reproducible example cases in a single place.

## Repository layout

- [src/](src/README.md) — C++ sources for the application (python bindings, app-specific code)
- [python/](python/README.md) — Python drivers/utilities (e.g., job management, ROM pipeline orchestration)
- [examples/](examples/README.md) — Example decks, runs, and reference cases

# Quickstart

## Dependencies

This repository builds a standalone application that links against a libROM installation, an OpenSn installation and certain OpenSn dependencies.

You must have the following available:

- OpenSn
- libROM
- OpenSn dependencies (VTK, Caliper, MPI, Python with pybind11)

---

## Build

From the repository root:

```bash
mkdir build
cd build
cmake .. \
  -DOpenSn_DIR=<path-to-opensn-install> \
  -DlibROM_DIR=<path-to-librom-install>
make -j
```
An example can be run from its respective folder with
```bash
python run_rom_*.py --exe=path/to/app/exe
```