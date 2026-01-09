// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "framework/parameters/input_parameters.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/lbs_problem.h"
#include "linalg/Matrix.h"
#include "linalg/Vector.h"
#include "algo/manifold_interp/MatrixInterpolator.h"
#include "algo/manifold_interp/VectorInterpolator.h"
#include "ROM_structs.h"
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

  void LoadUgs();

  /// Calculate AU via sweeps
  std::shared_ptr<CAROM::Matrix> AssembleAU();

  /// Sweep to form RHS
  std::shared_ptr<CAROM::Vector> AssembleRHS();

  /// Sweep to form BU
  std::shared_ptr<CAROM::Matrix> AssembleBU();

  /// Assemble the reduced system and save to file
  void AssembleROM(std::shared_ptr<CAROM::Matrix>& AU_, 
                   std::shared_ptr<CAROM::Vector>& b_, 
                   const std::string& Ar_filename,
                   const std::string& rhs_filename);

  /// Assemble the reduced system and save to file
  void AssembleROM(std::shared_ptr<CAROM::Matrix>& AU_, 
                   std::shared_ptr<CAROM::Matrix>& BU_, 
                   const std::string& Ar_filename,
                   const std::string& Br_filename);

  /// Solve given LHS and RHS of a ROM system
  void SolveROM(std::shared_ptr<CAROM::Matrix>& Ar,
                std::shared_ptr<CAROM::Vector>& rhs);
  
  /// Solve given LHS and RHS of a k-eigenvalue ROM system
  double SolveROM(std::shared_ptr<CAROM::Matrix>& Ar,
                  std::shared_ptr<CAROM::Matrix>& Br);

  /// Load Ar and initialize libROM interpolator objects
  void SetupArInterpolator(CAROM::Vector& desired_point);

  /// Load br and initialize libROM interpolator objects
  void SetuprhsInterpolator(CAROM::Vector& desired_point);

  /// Load Br and initialize libROM interpolator objects
  void SetupBrInterpolator(CAROM::Vector& desired_point);

  void InterpolateArAndRHS(
    CAROM::Vector& desired_point,
    std::shared_ptr<CAROM::Matrix>& Ar_interp,
    std::shared_ptr<CAROM::Vector>& rhs_interp);

  void InterpolateArAndBr(
    CAROM::Vector& desired_point,
    std::shared_ptr<CAROM::Matrix>& Ar_interp,
    std::shared_ptr<CAROM::Matrix>& Br_interp);

protected:
  std::unique_ptr<CAROM::Matrix> spatialbasis;
  std::vector<std::unique_ptr<CAROM::Matrix>> Ugs;
  std::unique_ptr<CAROM::MatrixInterpolator> Ar_interp_obj_ptr;
  std::unique_ptr<CAROM::MatrixInterpolator> Br_interp_obj_ptr;
  std::unique_ptr<CAROM::VectorInterpolator> rhs_interp_obj_ptr;

  std::shared_ptr<LBSProblem> lbs_problem;
  ROMOptions options_;

public:
  std::vector<CAROM::Vector> param_points_;
  int romRank;
};

} // namespace opensn
