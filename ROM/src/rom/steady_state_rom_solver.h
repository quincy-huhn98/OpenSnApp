// SPDX-FileCopyrightText: 2025 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/solvers/steady_state_solver.h"
#include "rom_problem.h"

namespace opensn
{

class LBSProblem;

class SteadyStateROMSolver : public SteadyStateSourceSolver
{
protected:
  std::shared_ptr<LBSProblem> lbs_problem_;
  std::shared_ptr<ROMProblem> rom_problem_;

public:
  explicit SteadyStateROMSolver(const InputParameters& params);

  void Initialize();

  void Execute();

public:
  static InputParameters GetInputParameters();

};

} // namespace opensn