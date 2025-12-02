// SPDX-FileCopyrightText: 2025 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/solvers/pi_keigen_solver.h"
#include "ROM_problem.h"

namespace opensn
{

class LBSProblem;

class PowerIterationKEigenROMSolver : public PowerIterationKEigenSolver
{
protected:  
  std::shared_ptr<LBSProblem> lbs_problem_;
  std::shared_ptr<ROMProblem> rom_problem_;

public:
  explicit PowerIterationKEigenROMSolver(const InputParameters& params);

  void Initialize() override;

  void Execute() override;

  static InputParameters GetInputParameters();
};

} // namespace opensn
