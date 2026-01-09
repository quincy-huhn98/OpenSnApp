# examples/

**Location:** [repo root](../README.md) / `examples/`

Example input decks and run scripts demonstrating how to use the application and ROM pipeline.

Each example folder typically contains:
- a base OpenSn input deck,
- a problem definition helper,
- a `run_rom_*.py` entry script you execute,
- optional reference data (e.g., base cross sections).

## Examples

- [`reed/`](reed/README.md)  
  Reed 1-D benchmark example.

- [`checkerboard/`](checkerboard/README.md)  
  2-D Checkerboard/lattice problem configuration.

- [`2gcheckerboard/`](2gcheckerboard/README.md)  
  Two-group checkerboard variant with absorber/scatterer material files.

- [`1dk/`](1dk/README.md)  
  1-D k-eigenvalue test problem with H2O/U material files.


## Running

Each example folder includes a `run_rom_*.py` script. Start with the folder README for details.

Execution pattern:

```bash
python run_rom_<example>.py --exe=path/to/app/exe
