// SPDX-FileCopyrightText: 2025 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "diff_py_app.h"
#include "mpicpp-lite/mpicpp-lite.h"
#include "petsc.h"

int
main(int argc, char** argv)
{
  try
  {
    int error_code = EXIT_FAILURE;
    bool petsc_initialized = false;
    {
      mpi::Environment env(argc, argv);
      py::scoped_interpreter guard{};
      diffpy::DiffApp app(MPI_COMM_WORLD);
      error_code = app.Run(argc, argv);
      petsc_initialized = app.IsPetscInitialized();
    }
    if (petsc_initialized)
      PetscFinalize();
    return error_code;
  }
  catch (const std::exception& e)
  {
    std::cerr << "Fatal error: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }
}
