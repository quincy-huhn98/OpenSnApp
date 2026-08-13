// SPDX-FileCopyrightText: 2025 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/discrete_ordinates_problem.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/solvers/steady_state_solver.h"
#include "rom_problem.h"

namespace opensn
{

class SteadyStateROMSolver : public SteadyStateSourceSolver
{
protected:
  std::shared_ptr<DiscreteOrdinatesProblem> lbs_problem_;
  std::shared_ptr<ROMProblem> rom_problem_;

public:
  /** Constructs a steady-state source ROM solver. */
  explicit SteadyStateROMSolver(const InputParameters& params);

  /** Initializes the steady-state ROM solver. */
  void Initialize();

  /** Executes the selected offline, merge, systems, MI-POD, or online ROM phase. */
  void Execute();

public:
  /** Returns the input-parameter schema for SteadyStateROMSolver. */
  static InputParameters GetInputParameters();
};

} // namespace opensn
