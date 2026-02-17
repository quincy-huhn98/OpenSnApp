// SPDX-FileCopyrightText: 2025 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "rom_py_app.h"
#include "petsc.h"

int main(int argc, char** argv)
{
  mpi::Environment env(argc, argv);

  PetscCall(PetscInitializeNoArguments()); // NOLINT(bugprone-casting-through-void)

  int retval = EXIT_SUCCESS;
  try
  {
    py::scoped_interpreter guard{};
    rompy::ROMApp app(MPI_COMM_WORLD); // NOLINT(bugprone-casting-through-void)
    retval = app.Run(argc, argv);
  }
  catch (...)
  {
    std::fprintf(stderr, "Unknown fatal error\n");
    retval = EXIT_FAILURE;
  }

  PetscFinalize();

  return retval;
}
