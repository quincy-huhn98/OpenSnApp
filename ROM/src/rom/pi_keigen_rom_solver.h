// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/discrete_ordinates_problem.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/solvers/pi_keigen_solver.h"
#include "rom_problem.h"

namespace opensn
{

class LBSProblem;

class PowerIterationKEigenROMSolver : public PowerIterationKEigenSolver
{
protected:
  std::shared_ptr<DiscreteOrdinatesProblem> lbs_problem_;
  std::shared_ptr<ROMProblem> rom_problem_;

public:
  /** Constructs a power-iteration k-eigenvalue ROM solver. */
  explicit PowerIterationKEigenROMSolver(const InputParameters& params);

  /** Initializes the underlying power-iteration solver. */
  void Initialize() override;

  /** Executes the selected offline, merge, systems, MI-POD, or online ROM phase. */
  void Execute() override;

  /** Returns the input-parameter schema for PowerIterationKEigenROMSolver. */
  static InputParameters GetInputParameters();
};

} // namespace opensn
