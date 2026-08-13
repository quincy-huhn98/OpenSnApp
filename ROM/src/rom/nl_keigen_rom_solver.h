// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/solver.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/discrete_ordinates_problem.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/solvers/nl_keigen_solver.h"
#include "rom_problem.h"

namespace opensn
{

class NLKEigenROMSolver : public Solver
{
protected:
  std::shared_ptr<DiscreteOrdinatesProblem> lbs_problem_;
  std::shared_ptr<ROMProblem> rom_problem_;
  NonLinearKEigenSolver nl_solver_;
  double k_eff_ = 1.0;
  bool initialized_ = false;

public:
  /** Constructs a nonlinear k-eigenvalue ROM solver. */
  explicit NLKEigenROMSolver(const InputParameters& params);

  /** Initializes the underlying nonlinear k-eigenvalue solver. */
  void Initialize() override;
  /** Executes the selected offline, merge, systems, MI-POD, or online ROM phase. */
  void Execute() override;

  /** Returns the most recently computed eigenvalue. */
  double GetEigenvalue() const;

  /** Returns the input-parameter schema for NLKEigenROMSolver. */
  static InputParameters GetInputParameters();
};

} // namespace opensn
