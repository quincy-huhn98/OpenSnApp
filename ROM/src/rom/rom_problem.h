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
  /** Constructs a ROM problem and attaches it to an existing discrete-ordinates problem. */
  explicit ROMProblem(const InputParameters& params);

  /** Returns the schema for the nested ROM options block. */
  static InputParameters GetOptionsBlock();
  /** Returns the input-parameter schema for ROMProblem. */
  static InputParameters GetInputParameters();
  /** Constructs a ROMProblem through the OpenSn object factory. */
  static std::shared_ptr<ROMProblem> Create(const ParameterBlock& params);

  /** Parses and stores ROM options from an input block. */
  void SetOptions(const InputParameters& input);
  /** Returns a reference to the solver options. */
  ROMOptions& GetOptions();

  /** Collects and writes one snapshot per energy group for snapshot identifier @p id. */
  void TakeSample(int id);

  /** Loads @p nsnaps snapshots per group and builds the group-wise spatial bases. */
  void MergePhase(int nsnaps);

  /** Reads a whitespace-delimited parameter matrix from @p filename. */
  void ReadParamMatrix(const std::string& filename);

  /** Loads the group-wise reduced bases from disk. */
  void LoadUgs();

  /** Assembles the full-order operator image AU used to form reduced systems. */
  std::shared_ptr<CAROM::Matrix> AssembleAU();

  /** Assembles the full-order right-hand-side vector. */
  std::shared_ptr<CAROM::Vector> AssembleRHS();

  /** Assembles the full-order production-operator image BU. */
  std::shared_ptr<CAROM::Matrix> AssembleBU();

  /** Forms and writes the reduced source system from @p AU and @p b. */
  void AssembleROM(std::shared_ptr<CAROM::Matrix>& AU,
                   std::shared_ptr<CAROM::Vector>& b,
                   const std::string& Ar_filename,
                   const std::string& rhs_filename);

  /** Forms and writes the reduced k-eigenvalue system from @p AU and @p BU. */
  void AssembleROM(std::shared_ptr<CAROM::Matrix>& AU,
                   std::shared_ptr<CAROM::Matrix>& BU,
                   const std::string& Ar_filename,
                   const std::string& Br_filename);

  /** Builds a minimally invasive basis, solves the source ROM, and reconstructs the state. */
  void MIPOD(std::shared_ptr<CAROM::Matrix>& Ar, std::shared_ptr<CAROM::Vector>& rhs);

  /** Builds a minimally invasive basis and solves the k-eigenvalue ROM. */
  double MIPOD(std::shared_ptr<CAROM::Matrix>& Ar, std::shared_ptr<CAROM::Matrix>& Br);

  /** Solves a reduced source system and reconstructs the full-order flux moments. */
  void SolveROM(std::shared_ptr<CAROM::Matrix>& Ar, std::shared_ptr<CAROM::Vector>& rhs);

  /** Solves a reduced k-eigenvalue system and reconstructs the full-order state. */
  double SolveROM(std::shared_ptr<CAROM::Matrix>& Ar, std::shared_ptr<CAROM::Matrix>& Br);

  /** Loads reduced loss matrices and initializes their interpolator at @p desired_point. */
  void SetupArInterpolator(CAROM::Vector& desired_point);

  /** Loads reduced source vectors and initializes their interpolator at @p desired_point. */
  void SetupRHSrInterpolator(CAROM::Vector& desired_point);

  /** Loads reduced production matrices and initializes their interpolator at @p desired_point. */
  void SetupBrInterpolator(CAROM::Vector& desired_point);

  /** Interpolates the reduced loss matrix and source vector at @p desired_point. */
  void InterpolateArAndRHSr(CAROM::Vector& desired_point,
                            std::shared_ptr<CAROM::Matrix>& Ar_interp,
                            std::shared_ptr<CAROM::Vector>& rhs_interp);

  /** Interpolates the reduced loss and production matrices at @p desired_point. */
  void InterpolateArAndBr(CAROM::Vector& desired_point,
                          std::shared_ptr<CAROM::Matrix>& Ar_interp,
                          std::shared_ptr<CAROM::Matrix>& Br_interp);

protected:
  std::unique_ptr<CAROM::Matrix> spatial_basis_;
  opensn::Vector<double> b_;
  std::vector<std::unique_ptr<CAROM::Matrix>> Ugs_;
  std::unique_ptr<CAROM::MatrixInterpolator> Ar_interp_obj_ptr_;
  std::unique_ptr<CAROM::MatrixInterpolator> Br_interp_obj_ptr_;
  std::unique_ptr<CAROM::VectorInterpolator> rhs_interp_obj_ptr_;

  std::shared_ptr<DiscreteOrdinatesProblem> lbs_problem_;
  ROMOptions options_;

public:
  std::vector<CAROM::Vector> param_points;
  int rom_rank;
};

} // namespace opensn
