// SPDX-FileCopyrightText: 2025 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "framework/parameters/input_parameters.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/discrete_ordinates_problem.h"
#include "linalg/Matrix.h"
#include "linalg/Vector.h"
#include "algo/manifold_interp/MatrixInterpolator.h"
#include "algo/manifold_interp/VectorInterpolator.h"
#include "rom_structs.h"
#include <memory>
#include <vector>

namespace opensn
{

class ROMProblem : public Problem
{
public:
  /// Input parameters based construction.
  explicit ROMProblem(const InputParameters& params);

  static InputParameters GetOptionsBlock();
  
  static InputParameters GetInputParameters();
  static std::shared_ptr<ROMProblem> Create(const ParameterBlock& params);

  void SetOptions(const InputParameters& input);
  /// Returns a reference to the solver options.
  ROMOptions& GetOptions();

  /// Save current Phi in the basis generator
  void TakeSample(int id);

  /// Load snapshots and perform SVD
  void MergePhase(int nsnaps);

  /// Load the params from file
  void ReadParamMatrix(const std::string& filename);

  /// Calculate AU via sweeps
  std::shared_ptr<CAROM::Matrix> AssembleAU();

  /// Sweep to form RHS
  std::shared_ptr<CAROM::Vector> AssembleRHS();

  /// Assemble the reduced system and save to file
  void AssembleROM(std::shared_ptr<CAROM::Matrix>& AU, 
                   std::shared_ptr<CAROM::Vector>& b, 
                   const std::string& Ar_filename,
                   const std::string& rhs_filename);

  /// Solve given LHS and RHS of a ROM system
  void SolveROM(std::shared_ptr<CAROM::Matrix>& Ar,
                std::shared_ptr<CAROM::Vector>& rhs);

  /// Load reduced systems and initialize libROM interpolator objects
  void SetupInterpolator(CAROM::Vector& desired_point);

  void InterpolateArAndRHSr(
    CAROM::Vector& desired_point,
    std::shared_ptr<CAROM::Matrix>& Ar_interp,
    std::shared_ptr<CAROM::Vector>& rhs_interp);

protected:
  std::unique_ptr<CAROM::Matrix> spatial_basis_;
  opensn::Vector<double> b_;
  std::unique_ptr<CAROM::MatrixInterpolator> Ar_interp_obj_ptr_;
  std::unique_ptr<CAROM::VectorInterpolator> rhs_interp_obj_ptr_;
  std::shared_ptr<DiscreteOrdinatesProblem> lbs_problem_;
  ROMOptions options_;

public:
  std::vector<CAROM::Vector> param_points;
  int rom_rank;
};

} // namespace opensn
