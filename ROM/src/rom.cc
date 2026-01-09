// SPDX-FileCopyrightText: 2025 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "py_wrappers.h"
#include "rom/rom_problem.h"
#include "rom/steady_state_rom_solver.h"
#include "rom/pi_keigen_rom_solver.h"
#include "modules/solver.h"
#include <memory>
#include <string>

namespace opensn
{
namespace py = pybind11;

template <class T>
static InputParameters KwargsToParams(const py::kwargs& kw)
{
  auto params = T::GetInputParameters();
  params.AssignParameters(kwargs_to_param_block(kw));
  return params;
}


// clang-format off
void WrapROM(py::module& m)
{
  // ROMProblem
  auto rom_problem = py::class_<ROMProblem, 
                     std::shared_ptr<ROMProblem>,
                     Problem>(
      m, "ROMProblem",
      R"(
      ROM controller for reduced-order modeling workflows.

      Wrapper of :cpp:class:`opensn::ROMProblem`.
      )");

  rom_problem.def(
      py::init([](py::kwargs kw)
      {
        auto params = ROMProblem::GetInputParameters();
        params.AssignParameters(kwargs_to_param_block(kw));
        return std::make_shared<ROMProblem>(params);
      }),
      R"(
      ROMProblem(**kwargs)

      Construct a ROMProblem directly from keyword arguments.
      )");

  rom_problem.def_static(
      "Create",
      [](py::kwargs kw)
      {
        return ROMProblem::Create(kwargs_to_param_block(kw));
      },
      R"(
      Create(**kwargs) -> ROMProblem

      Factory constructor (recommended). Accepts the same kwargs as the direct constructor:
        - problem : LBSProblem (required)
        - options : dict (optional)
        - name    : str (optional)
      )");


  // SteadyStateROMSolver
  auto ss_rom_solver =
      py::class_<SteadyStateROMSolver,
                 std::shared_ptr<SteadyStateROMSolver>,
                 Solver>(
        m,
        "SteadyStateROMSolver",
        R"(
        Steady-state ROM driver.

        Wrapper of :cpp:class:`opensn::SteadyStateROMSolver`.

        Parameters (kwargs)
        -------------------
        problem : LBSProblem
            The full-order transport problem.
        rom_problem : ROMProblem
            The ROM controller (bases, reduced systems, interpolation).
        name : str
            Required solver name for logging/monitors.
        )"
      );

  ss_rom_solver.def(
      py::init([](py::kwargs kw)
      {
        auto params = KwargsToParams<SteadyStateROMSolver>(kw);

        return std::make_shared<SteadyStateROMSolver>(params);
      }),
      R"(
      SteadyStateROMSolver(**kwargs)

      Construct a steady-state driver that dispatches to ROM or FOM paths
      depending on the ROM options and phase.
      )"
  );

  ss_rom_solver
      .def("Initialize", &SteadyStateROMSolver::Initialize,
           R"(
           Initialize()

           Prepare the solver and ROM controller for execution.
           )")
      .def("Execute",    &SteadyStateROMSolver::Execute,
           R"(
           Execute()

           Run the solve. Behavior depends on the ROM phase:
             - 'offline' : full-order solve + snapshot sample
             - 'merge'   : merge snapshots into bases
             - 'systems' : assemble reduced systems and write libROM files
             - 'online'  : interpolate and solve reduced system
           )");

  // PowerIterationKEigenROMSolver
  auto pi_rom_solver =
      py::class_<PowerIterationKEigenROMSolver,
                 std::shared_ptr<PowerIterationKEigenROMSolver>,
                 Solver>(
        m,
        "PowerIterationROMSolver",
        R"(
        k eigen ROM driver.

        Wrapper of :cpp:class:`opensn::PowerIterationKEigenROMSolver`.

        Parameters (kwargs)
        -------------------
        problem : LBSProblem
            The full-order transport problem.
        rom_problem : ROMProblem
            The ROM controller (bases, reduced systems, interpolation).
        name : str
            Required solver name for logging/monitors.
        )"
      );

  pi_rom_solver.def(
      py::init([](py::kwargs kw)
      {
        auto params = KwargsToParams<PowerIterationKEigenROMSolver>(kw);

        return std::make_shared<PowerIterationKEigenROMSolver>(params);
      }),
      R"(
      PowerIterationKEigenROMSolver(**kwargs)

      Construct a k-eigen driver that dispatches to ROM or FOM paths
      depending on the ROM options and phase.
      )"
  );

  pi_rom_solver
      .def("Initialize", &PowerIterationKEigenROMSolver::Initialize,
           R"(
           Initialize()

           Prepare the solver and ROM controller for execution.
           )")
      .def("Execute",    &PowerIterationKEigenROMSolver::Execute,
           R"(
           Execute()

           Run the solve. Behavior depends on the ROM phase:
             - 'offline' : full-order solve + snapshot sample
             - 'merge'   : merge snapshots into bases
             - 'systems' : assemble reduced systems and write libROM files
             - 'online'  : interpolate and solve reduced system
           )")
      .def("GetEigenvalue",
            &PowerIterationKEigenROMSolver::GetEigenvalue,
            R"(
            Return the current k‑eigenvalue.
            )");
}
// clang-format on


void py_rom(py::module& pyopensn)
{
  auto rom = pyopensn.def_submodule("rom", "ROM module.");
  WrapROM(rom);
}

} // namespace opensn