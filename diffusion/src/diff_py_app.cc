// SPDX-FileCopyrightText: 2025 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "diff_py_app.h"
#include "py_wrappers.h"
#include "opensn/python/lib/console.h"
#include "opensn/python/lib/py_wrappers.h"
#include "opensn/framework/logging/log.h"
#include "opensn/framework/utils/utils.h"
#include "opensn/framework/utils/timer.h"
#include "opensn/framework/runtime.h"
#include "caliper/cali.h"
#include "cxxopts/cxxopts.h"
#include "petsc.h"
#include <string>

using namespace opensn;
namespace py = pybind11;

namespace diffpy
{

DiffApp::DiffApp(const mpi::Communicator& comm) : allow_petsc_error_handler_(false)
{
  py::gil_scoped_acquire gil;
  opensn::mpi_comm = comm;

  py::module sys = py::module::import("sys");
  py::exec("rank = " + std::to_string(comm.rank()));
  py::exec("size = " + std::to_string(comm.size()));
  py::exec("opensn_console = True");

  opensnpy::console.BindBarrier(comm);
  opensnpy::console.BindAllReduce(comm);

  opensnpy::Console::BindModule(WrapYlm);
  opensnpy::Console::BindModule(WrapVector3);
  opensnpy::Console::BindModule(WrapFunctors);

  opensnpy::Console::BindModule(WrapQuadraturePointPhiTheta);
  opensnpy::Console::BindModule(WrapQuadrature);
  opensnpy::Console::BindModule(WrapProductQuadrature);
  opensnpy::Console::BindModule(WrapTriangularQuadrature);
  opensnpy::Console::BindModule(WrapCurvilinearProductQuadrature);
  opensnpy::Console::BindModule(WrapSLDFEsqQuadrature);
  opensnpy::Console::BindModule(WrapLebedevQuadrature);

  opensnpy::Console::BindModule(WrapMesh);
  opensnpy::Console::BindModule(WrapMeshGenerator);
  opensnpy::Console::BindModule(WrapGraphPartitioner);

  opensnpy::Console::BindModule(WrapLogicalVolume);

  opensnpy::Console::BindModule(WrapPointSource);
  opensnpy::Console::BindModule(WrapVolumetricSource);

  opensnpy::Console::BindModule(WrapMultiGroupXS);

  opensnpy::Console::BindModule(WrapFieldFunction);
  opensnpy::Console::BindModule(WrapFieldFunctionGridBased);
  opensnpy::Console::BindModule(WrapFieldFunctionInterpolation);

  opensnpy::Console::BindModule(WrapResEval);

  opensnpy::Console::BindModule(WrapProblem);
  opensnpy::Console::BindModule(WrapSolver);
  opensnpy::Console::BindModule(WrapLBS);
  opensnpy::Console::BindModule(WrapSteadyState);
  opensnpy::Console::BindModule(WrapTransient);
  opensnpy::Console::BindModule(WrapNLKEigen);
  opensnpy::Console::BindModule(WrapPIteration);
  opensnpy::Console::BindModule(WrapDiscreteOrdinatesKEigenAcceleration);

  opensnpy::Console::BindModule(WrapDiffusion);

  // PostProcessor/Printer wrappers removed in current OpenSn headers.
}

int
DiffApp::InitPETSc(int argc, char** argv)
{
  (void)argc;
  (void)argv;
  PetscOptionsInsertString(nullptr, "-error_output_stderr");
  if (!allow_petsc_error_handler_)
    PetscOptionsInsertString(nullptr, "-no_signal_handler");
  // Avoid parsing command-line options; we manage options explicitly.
  PetscCall(PetscInitialize(nullptr, nullptr, nullptr, nullptr));
  petsc_initialized_ = true;
  return 0;
}

int
DiffApp::Run(int argc, char** argv)
{
  if (opensn::mpi_comm.rank() == 0)
  {
    std::cout << opensn::program << " version " << GetVersionStr() << "\n"
              << Timer::GetLocalDateTimeString() << " Running " << opensn::program << " with "
              << opensn::mpi_comm.size() << " processes.\n"
              << opensn::program << " number of arguments supplied: " << argc - 1 << "\n"
              << std::endl;
  }

  if (ProcessArguments(argc, argv))
  {
    InitPETSc(argc, argv);

    opensn::Initialize();
    opensnpy::console.InitConsole();
    opensnpy::console.ExecuteFile(opensn::input_path.string());
    opensn::Finalize();

    if (opensn::mpi_comm.rank() == 0)
    {
      std::cout << "\nElapsed execution time: " << program_timer.GetTimeString() << "\n"
                << Timer::GetLocalDateTimeString() << " " << opensn::program
                << " finished execution." << std::endl;
    }
  }
  else
    return help_requested_ ? EXIT_SUCCESS : EXIT_FAILURE;

  cali_mgr.flush();
  return EXIT_SUCCESS;
}

bool
DiffApp::ProcessArguments(int argc, char** argv)
{
  cxxopts::Options options(LowerCase(opensn::program), "");

  try
  {
    /* clang-format off */
    options.add_options("User")
    ("h,help",                      "Help message")
    ("c,suppress-color",            "Suppress color output")
    ("v,verbose",                   "Verbosity level (0 to 3). Default is 0.", cxxopts::value<int>())
    ("caliper",                     "Enable Caliper reporting",
      cxxopts::value<std::string>()->implicit_value("runtime-report(calc.inclusive=true),max_column_width=80"))
    ("i,input",                     "Input file", cxxopts::value<std::string>())
    ("allow-petsc-error-handler",   "Allow PETSc error handler")
    ("p,py",                        "Python expression", cxxopts::value<std::vector<std::string>>());
    /* clang-format on */

    auto result = options.parse(argc, argv);

    if (result.count("help"))
    {
      if (opensn::mpi_comm.rank() == 0)
        std::cout << options.help({"User"}) << std::endl;
      help_requested_ = true;
      return false;
    }

    if (result.count("verbose"))
    {
      int verbosity = result["verbose"].as<int>();
      opensn::log.SetVerbosity(verbosity);
    }

    if (result.count("allow-petsc-error-handler"))
      allow_petsc_error_handler_ = true;

    if (result.count("suppress-color"))
      opensn::suppress_color = true;

    if (result.count("caliper"))
    {
      opensn::use_caliper = true;
      opensn::cali_config = result["caliper"].as<std::string>();
    }

    if (result.count("py"))
    {
      for (const auto& pyarg : result["py"].as<std::vector<std::string>>())
        opensnpy::console.GetCommandBuffer().push_back(pyarg);
    }

    opensn::input_path = result["input"].as<std::string>();
    if (not std::filesystem::exists(input_path) or not std::filesystem::is_regular_file(input_path))
    {
      if (opensn::mpi_comm.rank() == 0)
        std::cerr << "Invalid input file: " << input_path.string() << "\n" << std::endl;
      return false;
    }
  }
  catch (const std::exception& e)
  {
    if (opensn::mpi_comm.rank() == 0)
      std::cerr << e.what() << "\n" << options.help({"User"}) << std::endl;
    return false;
  }

  return true;
}

} // namespace diffpy
