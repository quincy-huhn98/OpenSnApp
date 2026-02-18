# src/

**Location:** [repo root](../README.md) / `src/`

C++ sources for the application. This folder contains the main entry point, ROM implementation, and the Python-app bridge used to integrate with OpenSn problems.

## Key files

- [`main.cc`](main.cc)  
  Application entry point.

- [`rom.cc`](rom.cc)  
  Top-level ROM plumbing/glue code.

- [`rom_py_app.h`](rom_py_app.h), [`rom_py_app.cc`](rom_py_app.cc)  
  ROM “Python app” integration layer (app interface invoked from Python-side workflows).

- [`py_wrappers.h`](py_wrappers.h)  
  C++ declarations for Python-facing wrappers (bindings/glue utilities).

- [`rom/`](rom/README.md)  
  ROM core implementation (problem definition, solver, structs).