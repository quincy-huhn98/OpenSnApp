#!/usr/bin/env python3
"""Helpers for running complete ROM example pipelines in isolated directories."""

import os
import shutil
import sys
import tempfile
import numpy as np
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def isolated_example(example_name):
    """Yield an isolated example tree and make its Python modules importable."""
    rom_root = Path(__file__).resolve().parents[2]
    source = rom_root / "examples" / example_name
    test_root = rom_root / "test" / "work"
    test_root.mkdir(exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix=f"rom-integration-{example_name}-", dir=test_root))
    workdir = root / example_name
    shutil.copytree(source, workdir)

    old_directory = Path.cwd()
    old_path = list(sys.path)
    sys.path[:0] = [str(workdir), str(rom_root / "python")]
    os.chdir(workdir)
    try:
        yield workdir
    finally:
        os.chdir(old_directory)
        sys.path[:] = old_path
        shutil.rmtree(root)


def rom_executable():
    """Return the executable selected by ROM/test/run_tests."""
    executable = os.environ.get("ROM_APP_EXEC")
    if not executable:
        raise RuntimeError("ROM_APP_EXEC is not set; invoke this test with ROM/test/run_tests")
    return executable


def relative_flux_error(fom_flux, rom_flux, normalize=False):
    """Return a sign-invariant relative error for grouped flux arrays."""
    fom_flux = np.asarray(fom_flux)
    rom_flux = np.asarray(rom_flux)
    if normalize:
        fom_flux = fom_flux / np.linalg.norm(fom_flux)
        rom_flux = rom_flux / np.linalg.norm(rom_flux)
        if np.vdot(fom_flux, rom_flux) < 0.0:
            rom_flux *= -1.0
    return np.linalg.norm(rom_flux - fom_flux) / np.linalg.norm(fom_flux)
