// SPDX-FileCopyrightText: 2025 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/discrete_ordinates_problem.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/iterative_methods/wgs_context.h"
#include "framework/object_factory.h"
#include "framework/logging/log.h"
#include "framework/runtime.h"
#include "rom_problem.h"
#include "linalg/BasisGenerator.h"
#include "linalg/BasisReader.h"
#include <fstream>
#include <memory>
#include <cassert>

namespace opensn
{

OpenSnRegisterObjectInNamespace(rom, ROMProblem);

/** Returns the input-parameter schema for ROMProblem.
 *
 * Extends the base Problem schema with:
 * - `problem` : an existing LBSProblem instance to control.
 * - `options` : nested parameter block controlling ROM workflow (phase, IDs, files, etc.).
 */
InputParameters
ROMProblem::GetInputParameters()
{
  InputParameters params = Problem::GetInputParameters();

  params.SetClassName("ROMProblem");

  params.ChangeExistingParamToOptional("name", "ROMProblem");

  params.AddRequiredParameter<std::shared_ptr<Problem>>(
      "problem", "An existing discrete ordinates problem to attach the ROM controller to.");

  // Optional nested ROM options block (phase, ids, files, new_point, etc.)
  params.AddOptionalParameterBlock(
      "options", ParameterBlock(), "ROM control options (phase, param_id, param_file, new_point, …)");

  return params;
}

/** Factory helper that constructs a ROMProblem via the OpenSn object factory.
 *
 * \param params Parameter block (typically produced by the input system).
 * \return Shared pointer to a constructed ROMProblem.
 */
std::shared_ptr<ROMProblem>
ROMProblem::Create(const ParameterBlock& params)
{
  auto& factory = opensn::ObjectFactory::GetInstance();
  return factory.Create<ROMProblem>("rom::ROMProblem", params);
}

/** Constructs a ROMProblem and attaches it to an existing LBSProblem.
 *
 * The `problem` parameter must refer to a valid LBSProblem instance.
 * If an `options` block is provided, it is validated against GetOptionsBlock()
 * and stored into the internal ROMOptions structure.
 */
ROMProblem::ROMProblem(const InputParameters& params)
  : Problem(params),
  lbs_problem_(params.GetSharedPtrParam<Problem, DiscreteOrdinatesProblem>("problem"))
{
  // Initialize options
  if (params.IsParameterValid("options"))
  {
    auto options_params = ROMProblem::GetOptionsBlock();
    options_params.AssignParameters(params.GetParam("options"));
    SetOptions(options_params);
  }
}

/** Collects and writes one snapshot per energy group for the current state.
 *
 * This routine extracts the current `phi_new` vector from the attached LBSProblem,
 * forms a per-group snapshot of size (local_nodes * moments), and writes it using
 * libROM's snapshot format.
 *
 * \param id Snapshot identifier appended to the snapshot filename.
 */
void
ROMProblem::TakeSample(int id)
{
  bool update_right_SV = false;
  int max_num_snapshots = 100;
  bool isIncremental = false;
  const std::string basisName = "basis/snapshots_";

  auto num_moments = lbs_problem_->GetNumMoments();
  auto num_groups = lbs_problem_->GetNumGroups();
  auto num_local_nodes  = lbs_problem_->GetLocalNodeCount();
  std::vector<double> phi_new_local = lbs_problem_->GetPhiNewLocal();

  for (int g = 0; g < num_groups; ++g)
  {
    const std::string basisFileName = basisName + std::to_string(g) + "_" + std::to_string(id);

    int group_dim = num_local_nodes * num_moments;

    CAROM::Options options(group_dim, max_num_snapshots, update_right_SV);
    CAROM::BasisGenerator generator(options, isIncremental, basisFileName);

    std::vector<double> phi_group(group_dim, 0.0);

    for (int n = 0; n < num_local_nodes; ++n)
    {
      size_t node_base_full  = n * num_moments * num_groups;
      size_t node_base_group = n * num_moments;

      for (int m = 0; m < num_moments; ++m)
      {
        auto idx_full  = node_base_full  + m * num_groups + g;
        auto idx_group = node_base_group + m;

        phi_group[idx_group] = phi_new_local[idx_full];
      }
    }

    generator.takeSample(phi_group.data());
    generator.writeSnapshot();
  }
}

/** Builds (merges) group-wise spatial bases from previously written snapshots.
 *
 * For each group, loads `nsnaps` snapshot files, performs an SVD-based basis
 * construction with a prescribed tolerance and maximum rank, and writes basis data.
 * Singular values are additionally dumped to text on rank 0 for diagnostics.
 *
 * \param nsnaps Number of snapshot IDs to load per group.
 */
void
ROMProblem::MergePhase(int nsnaps)
{
  bool update_right_SV = false;
  int max_num_snapshots = 300;
  bool isIncremental = false;

  auto num_moments = lbs_problem_->GetNumMoments();
  auto num_groups = lbs_problem_->GetNumGroups();
  auto num_local_nodes  = lbs_problem_->GetLocalNodeCount();
  auto group_dim = num_local_nodes * num_moments;
  auto full_dim = num_local_nodes * num_moments * num_groups;

  CAROM::Options options(group_dim, max_num_snapshots, update_right_SV);
  double tol = 1e-8;
  rom_rank = 20;
  options.setSingularValueTol(tol);
  options.setMaxBasisDimension(rom_rank);

  for (auto g = 0; g < num_groups; ++g)
  {
    auto basis_prefix = "basis/basis_" + std::to_string(g);
    CAROM::BasisGenerator loader(options, isIncremental, basis_prefix);

    for (auto paramID = 0; paramID < nsnaps; ++paramID)
    {
      auto snap_file = "basis/snapshots_" + std::to_string(g) + "_" + std::to_string(paramID) + "_snapshot";
      loader.loadSamples(snap_file, "snapshot");
    }

    loader.endSamples();

    // Save singular values per group
    if (opensn::mpi_comm.rank() == 0)
    {
      auto sv_file = "data/singular_values_g" + std::to_string(g) + ".txt";
      std::ofstream sv_out(sv_file);
      auto S_vec = loader.getSingularValues();
      for (auto i = 0; i < S_vec->dim(); ++i)
        sv_out << std::setprecision(16) << S_vec->item(i) << "\n";
      sv_out.close();
      log.Log() << "Saved singular values for group " << g << " to " << sv_file << "\n";
    }
  }
}

/** Reads a whitespace-delimited parameter matrix from a text file.
 *
 * Each non-empty line becomes one parameter point. Points are stored as libROM
 * vectors in \c param_points.
 *
 * \param filename Path to the parameter matrix text file.
 */
void
ROMProblem::ReadParamMatrix(const std::string& filename)
{
  param_points.clear();

  std::ifstream infile(filename);
  std::string line;

  while (std::getline(infile, line))
  {
    std::istringstream iss(line);
    std::vector<double> row;
    double val;
    while (iss >> val)
      row.push_back(val);

    if (!row.empty())
      param_points.emplace_back(row.data(), static_cast<int>(row.size()),false,true);
  }
}

/** Loads the group-wise reduced bases from disk.
 *
 * Each group basis is read from the corresponding libROM basis file and
 * stored in \c Ugs_. The reduced dimension is inferred from the first
 * basis and stored in \c rom_rank.
 */
void
ROMProblem::LoadUgs()
{
  auto num_groups = lbs_problem_->GetNumGroups();
  Ugs_.reserve(num_groups);
  for (auto g = 0; g < num_groups; ++g)
  {
    const auto basis_root = "basis/basis_" + std::to_string(g);
    auto reader = std::make_unique<CAROM::BasisReader>(basis_root);
    auto Ug = reader->getSpatialBasis();
    if (g == 0) rom_rank = Ug->numColumns();
    Ugs_.push_back(std::move(Ug));
  }
}

/** Assembles the global (local-DOF) matrix AU used to form reduced systems.
 *
 * For each group \c g and basis vector \c r, this method:
 * 1. Injects the basis column into the full-order dof layout.
 * 2. Uses the LBS WGS context to perform a sweep with that vector as a source.
 * 3. Forms the corresponding column of AU as (injected_basis - resulting_phi_new).
 *
 * \return Shared pointer to AU of size (local_dofs) x (rom_rank * num_groups).
 */
std::shared_ptr<CAROM::Matrix>
ROMProblem::AssembleAU()
{
  auto num_moments = lbs_problem_->GetNumMoments();
  auto num_groups = lbs_problem_->GetNumGroups();
  auto num_local_nodes  = lbs_problem_->GetLocalNodeCount();
  const auto num_local_dofs = num_local_nodes * num_moments * num_groups;

  std::vector<std::unique_ptr<CAROM::Matrix>> Ugs;
  Ugs.reserve(num_groups);
  for (auto g = 0; g < num_groups; ++g)
  {
    const auto basis_root = "basis/basis_" + std::to_string(g);
    auto reader = std::make_unique<CAROM::BasisReader>(basis_root);
    auto Ug = reader->getSpatialBasis();
    if (g == 0) rom_rank = Ug->numColumns();
    Ugs.push_back(std::move(Ug));
  }

  auto AU = std::make_shared<CAROM::Matrix>(num_local_dofs, rom_rank * num_groups, true);

  // Assuming one groupset for ROM problems
  assert(lbs_problem_->GetNumWGSSolvers() == 1); 
  auto raw_context = lbs_problem_->GetWGSSolver(0)->GetContext();
  auto gs_context    = std::dynamic_pointer_cast<WGSContext>(raw_context);
  const auto scope   = gs_context->lhs_src_scope;

  auto& phi_old_local = lbs_problem_->GetPhiOldLocal();
  auto& q_moments_local = lbs_problem_->GetQMomentsLocal();
  
  for (auto g = 0; g < num_groups; ++g)
  {
    for (auto r = 0; r < rom_rank; ++r)
    {
      std::vector<double> basis_local(num_local_dofs, 0.0);
      phi_old_local.assign(phi_old_local.size(), 0.0);

      auto col_g  = Ugs[g]->getColumn(r);
      size_t rowg = 0;
      for (size_t n = 0; n < num_local_nodes; ++n)
        for (size_t m = 0; m < static_cast<size_t>(num_moments); ++m, ++rowg)
        {
          const size_t row_phi = n * (num_moments * num_groups) + m * num_groups + g;
          basis_local[row_phi] = col_g->item(rowg);
        }

      q_moments_local.assign(q_moments_local.size(), 0.0);
      gs_context->set_source_function(gs_context->groupset, q_moments_local, basis_local, scope);

      // Sweep
      gs_context->ApplyInverseTransportOperator(scope);
      std::vector<double> phi_new_local = lbs_problem_->GetPhiNewLocal();

      const auto col_idx = g * rom_rank + r;
      for (size_t i = 0; i < static_cast<size_t>(num_local_dofs); ++i)
        AU->item(i, static_cast<int>(col_idx)) = basis_local[i] - phi_new_local[i];
    }
  }
  return AU;
}

/** Assembles the full-order right-hand-side vector used for ROM system build.
 *
 * The RHS corresponds to the sweep result for the current "old" iterate/source
 * configuration and is used with AU to form the reduced system:
 * rhs = AU^T b.
 *
 * \return Shared pointer to b of length local_dofs.
 */
std::shared_ptr<CAROM::Vector> 
ROMProblem::AssembleRHS()
{
  auto num_moments = lbs_problem_->GetNumMoments();
  auto num_groups = lbs_problem_->GetNumGroups();
  auto num_local_nodes = lbs_problem_->GetLocalNodeCount();
  auto num_local_dofs = num_local_nodes * num_moments * num_groups;
  auto b = std::make_shared<CAROM::Vector>(num_local_dofs, true);

  // Assuming one groupset for ROM problems
  assert(lbs_problem_->GetNumWGSSolvers() == 1);
  auto raw_context = lbs_problem_->GetWGSSolver(0)->GetContext();
  auto gs_context_ptr = std::dynamic_pointer_cast<WGSContext>(raw_context);
  auto scope = gs_context_ptr->rhs_src_scope;

  auto& phi_old_local = lbs_problem_->GetPhiOldLocal();
  auto& q_moments_local = lbs_problem_->GetQMomentsLocal();

  q_moments_local.assign(q_moments_local.size(), 0.0);
  gs_context_ptr->set_source_function(gs_context_ptr->groupset, q_moments_local, phi_old_local, scope);

  // Sweep
  gs_context_ptr->ApplyInverseTransportOperator(scope);
  std::vector<double> phi_new_local = lbs_problem_->GetPhiNewLocal();

  for (int i = 0; i < num_local_dofs; ++i)
    (*b)(i) = phi_new_local[i];
  return b;
}

/** Assembles the full-order operator image BU for k-eigenvalue ROM systems.
 *
 * For each group \c g and basis vector \c r, this method injects the basis
 * column into the full-order dof layout, applies the inverse transport
 * operator with fission, scattering, and fixed sources enabled, and stores
 * the resulting vector as a column of BU.
 *
 * \return Shared pointer to BU of size (local_dofs) x (rom_rank * num_groups).
 */
std::shared_ptr<CAROM::Matrix>
ROMProblem::AssembleBU()
{
  auto num_moments = lbs_problem_->GetNumMoments();
  auto num_groups = lbs_problem_->GetNumGroups();
  auto num_local_nodes  = lbs_problem_->GetLocalNodeCount();
  const auto num_local_dofs = num_local_nodes * num_moments * num_groups;

  auto BU = std::make_shared<CAROM::Matrix>(num_local_dofs, rom_rank * num_groups, true);

  // Assuming one groupset for ROM problems
  assert(lbs_problem_->GetNumWGSSolvers() == 1);
  auto raw_context = lbs_problem_->GetWGSSolver(0)->GetContext();
  auto gs_context    = std::dynamic_pointer_cast<WGSContext>(raw_context);
  const auto scope   = APPLY_AGS_FISSION_SOURCES | APPLY_WGS_FISSION_SOURCES | APPLY_AGS_SCATTER_SOURCES | APPLY_FIXED_SOURCES;

  auto& phi_old_local = lbs_problem_->GetPhiOldLocal();
  auto& q_moments_local = lbs_problem_->GetQMomentsLocal();
  
  for (auto g = 0; g < num_groups; ++g)
  {
    for (auto r = 0; r < rom_rank; ++r)
    {
      std::vector<double> basis_local(num_local_dofs, 0.0);
      phi_old_local.assign(phi_old_local.size(), 0.0);

      auto col_g  = Ugs_[g]->getColumn(r);
      size_t rowg = 0;
      for (size_t n = 0; n < num_local_nodes; ++n)
        for (size_t m = 0; m < static_cast<size_t>(num_moments); ++m, ++rowg)
        {
          const size_t row_phi = n * (num_moments * num_groups) + m * num_groups + g;
          basis_local[row_phi] = col_g->item(rowg);
        }

      q_moments_local.assign(q_moments_local.size(), 0.0);
      gs_context->set_source_function(gs_context->groupset, q_moments_local, basis_local, scope);

      // Sweep
      gs_context->ApplyInverseTransportOperator(scope);
      std::vector<double> phi_new_local = lbs_problem_->GetPhiNewLocal();

      const auto col_idx = g * rom_rank + r;
      for (size_t i = 0; i < static_cast<size_t>(num_local_dofs); ++i)
        BU->item(i, static_cast<int>(col_idx)) = phi_new_local[i];
    }
  }
  return BU;
}

/** Forms and writes the reduced system (Ar, rhs) for a given AU and b.
 *
 * Computes:
 * - rhs = AU^T * b
 * - Ar  = AU^T * AU
 * and writes them to libROM files.
 *
 * \param AU Full-order operator matrix assembled by AssembleAU().
 * \param b  Full-order RHS vector assembled by AssembleRHS().
 * \param Ar_filename Output filename for Ar.
 * \param rhs_filename Output filename for rhs.
 */
void 
ROMProblem::AssembleROM(
  std::shared_ptr<CAROM::Matrix>& AU,
  std::shared_ptr<CAROM::Vector>& b,
  const std::string& Ar_filename,
  const std::string& rhs_filename)
{
  // rhs = AU^T * b
  auto rhs = AU->transposeMult(*b);

  // Ar = AU^T * AU
  auto Ar = AU->transposeMult(*AU);

  // Save
  Ar->write(Ar_filename);
  rhs->write(rhs_filename);
}

/** Forms and writes the reduced matrices (Ar, Br) for a k-eigenvalue ROM.
 *
 * Computes:
 * - Br = AU^T * BU
 * - Ar = AU^T * AU
 * and writes both matrices to libROM files.
 *
 * \param AU Full-order operator matrix assembled by AssembleAU().
 * \param BU Full-order operator image assembled by AssembleBU().
 * \param Ar_filename Output filename for Ar.
 * \param Br_filename Output filename for Br.
 */
void 
ROMProblem::AssembleROM(
  std::shared_ptr<CAROM::Matrix>& AU,
  std::shared_ptr<CAROM::Matrix>& BU,
  const std::string& Ar_filename,
  const std::string& Br_filename)
{
  // Br = AU^T * BU
  auto Br = AU->transposeMult(*BU);

  // Ar = AU^T * AU
  auto Ar = AU->transposeMult(*AU);

  // Save
  Ar->write(Ar_filename);
  Br->write(Br_filename);
}

/** Solves the reduced system and reconstructs the full-order flux moments.
 *
 * Solves Ar * c = rhs by explicit inversion (Ar^{-1} rhs), then reconstructs
 * the full-order local dof vector using the stored group-wise bases:
 *   phi_new += sum_{g,r} c_{g,r} * U_g(:,r) injected into full layout.
 *
 * \param Ar Reduced matrix (rom_dim x rom_dim).
 * \param rhs Reduced RHS vector (rom_dim).
 */
void
ROMProblem::SolveROM(
  std::shared_ptr<CAROM::Matrix>& Ar,
  std::shared_ptr<CAROM::Vector>& rhs)
{
  auto Ar_inv = std::make_shared<CAROM::Matrix>(Ar->numRows(), Ar->numColumns(), false);

  Ar->inverse(*Ar_inv);

  auto c_vec = Ar_inv->mult(*rhs);

  auto num_moments = lbs_problem_->GetNumMoments();
  auto num_groups = lbs_problem_->GetNumGroups();
  auto num_local_nodes = lbs_problem_->GetLocalNodeCount();
  auto num_local_dofs = num_local_nodes * num_moments * num_groups;

  auto& phi_new_local = lbs_problem_->GetPhiNewLocal();
  phi_new_local.assign(phi_new_local.size(), 0.0);

  for (int g = 0; g < num_groups; ++g)
  {
    for (int r = 0; r < rom_rank; ++r)
    {
      const int cr_idx = g * rom_rank + r;
      const double cr  = (*c_vec)(cr_idx);

      auto col_g = Ugs_[g]->getColumn(r);
      size_t row_g = 0;
      for (size_t n = 0; n < num_local_nodes; ++n)
        for (size_t m = 0; m < static_cast<size_t>(num_moments); ++m, ++row_g)
        {
          const size_t row_phi = n * (num_moments * num_groups) + m * num_groups + static_cast<size_t>(g);
          phi_new_local[row_phi] += cr * col_g->item(row_g);
        }
    }
  }
}


/** Solves the reduced k-eigenvalue problem and reconstructs the full-order state.
 *
 * Forms the reduced operator Ar^{-1} Br, computes its right eigenpairs,
 * selects the dominant positive real eigenvalue, and reconstructs the
 * associated full-order local flux moments using the stored group-wise bases.
 *
 * \param Ar Reduced loss/operator matrix.
 * \param Br Reduced production/operator matrix.
 * \return The dominant positive real eigenvalue.
 */
double
ROMProblem::SolveROM(
  std::shared_ptr<CAROM::Matrix>& Ar,
  std::shared_ptr<CAROM::Matrix>& Br)
{
  auto Ar_inv = std::make_shared<CAROM::Matrix>(Ar->numRows(), Ar->numColumns(), false);
  auto Ar_inv_Br = std::make_shared<CAROM::Matrix>(Ar->numRows(), Ar->numColumns(), false);

  Ar->inverse(*Ar_inv);

  Ar_inv_Br = Ar_inv->mult(*Br);

  auto eigen_pair = CAROM::NonSymmetricRightEigenSolve(*Ar_inv_Br);

  double k_eff = 0.0;
  int best_col = -1;

  for (int i = 0; i < (int)eigen_pair.eigs.size(); ++i)
  {
    const auto& lam = eigen_pair.eigs[i];
    if (std::abs(lam.imag()) > 1.0e-10) continue;
    if (lam.real() <= 0.0) continue;
    if (lam.real() > k_eff)
    {
      k_eff = lam.real();
      best_col = i;
    }
  }

  auto num_moments = lbs_problem_->GetNumMoments();
  auto num_groups = lbs_problem_->GetNumGroups();
  auto num_local_nodes = lbs_problem_->GetLocalNodeCount();
  auto num_local_dofs = num_local_nodes * num_moments * num_groups;

  auto& phi_new_local = lbs_problem_->GetPhiNewLocal();
  phi_new_local.assign(phi_new_local.size(), 0.0);

  for (int g = 0; g < num_groups; ++g)
  {
    for (int r = 0; r < rom_rank; ++r)
    {
      const int cr_idx = g * rom_rank + r;
      const double cr  = eigen_pair.ev_real->item(cr_idx, best_col);

      auto col_g = Ugs_[g]->getColumn(r);
      size_t row_g = 0;
      for (size_t n = 0; n < (size_t)num_local_nodes; ++n)
        for (size_t m = 0; m < (size_t)num_moments; ++m, ++row_g)
        {
          const size_t row_phi =
            n * ((size_t)num_moments * (size_t)num_groups) + m * (size_t)num_groups + (size_t)g;
          phi_new_local[row_phi] += cr * col_g->item(row_g);
        }
    }
  }
  return k_eff;
}

/** Loads reduced matrices Ar and initializes the libROM matrix interpolator.
 *
 * One reduced matrix is read for each sampled parameter point. Identity
 * rotations are constructed, the closest sampled point to \p desired_point
 * is used as the reference index, and the interpolator is initialized for
 * subsequent online interpolation of Ar.
 *
 * \param desired_point Online parameter point used to choose the reference sample.
 */
void
ROMProblem::SetupArInterpolator(CAROM::Vector& desired_point)
{
  std::vector<std::shared_ptr<CAROM::Matrix>> Ar_matrices;

  // Load Ar and rhs from libROM files
  for (size_t i = 0; i < param_points.size(); ++i)
  {
    const std::string Ar_filename = "data/rom_system_Ar_" + std::to_string(i);

    // Create empty containers
    auto Ar = std::make_shared<CAROM::Matrix>();

    // Read matrix
    Ar->local_read(Ar_filename, opensn::mpi_comm.rank());
    Ar_matrices.push_back(Ar);
  }

  // Make Identity Rotations
  std::vector<std::shared_ptr<CAROM::Matrix>> rotations;
  int rom_dim = Ar_matrices[0]->numRows();

  for (size_t i = 0; i < Ar_matrices.size(); ++i)
  {
    std::shared_ptr<CAROM::Matrix> I;
    I = std::make_shared<CAROM::Matrix>(rom_dim, rom_dim, false);
    for (int j = 0; j < rom_dim; ++j)
      I->item(j, j) = 1.0;
    rotations.push_back(I);
  }

  int ref_index = getClosestPoint(param_points, desired_point);

  Ar_interp_obj_ptr_ = std::make_unique<CAROM::MatrixInterpolator>(
    param_points, rotations, Ar_matrices,
    ref_index, "SPD", "G", "LS", 0.999, false);
}

/** Loads reduced RHS vectors and initializes the libROM vector interpolator.
 *
 * One reduced RHS vector is read for each sampled parameter point. Identity
 * rotations are constructed, the closest sampled point to \p desired_point
 * is used as the reference index, and the interpolator is initialized for
 * subsequent online interpolation of the reduced RHS.
 *
 * \param desired_point Online parameter point used to choose the reference sample.
 */
void
ROMProblem::SetupRHSrInterpolator(CAROM::Vector& desired_point)
{
  std::vector<std::shared_ptr<CAROM::Vector>> rhs_vectors;

  // Load Ar and rhs from libROM files
  for (size_t i = 0; i < param_points.size(); ++i)
  {
    const std::string rhs_filename = "data/rom_system_br_" + std::to_string(i);

    // Create empty containers
    auto rhs = std::make_shared<CAROM::Vector>();

    // Read vector
    rhs->local_read(rhs_filename, opensn::mpi_comm.rank());
    rhs_vectors.push_back(rhs);
  }

  // Make Identity Rotations
  std::vector<std::shared_ptr<CAROM::Matrix>> rotations;
  int rom_dim = rhs_vectors[0]->dim();

  for (size_t i = 0; i < rhs_vectors.size(); ++i)
  {
    std::shared_ptr<CAROM::Matrix> I;
    I = std::make_shared<CAROM::Matrix>(rom_dim, rom_dim, false);
    for (int j = 0; j < rom_dim; ++j)
      I->item(j, j) = 1.0;
    rotations.push_back(I);
  }

  int ref_index = getClosestPoint(param_points, desired_point);

  rhs_interp_obj_ptr_ = std::make_unique<CAROM::VectorInterpolator>(
    param_points, rotations, rhs_vectors,
    ref_index, "G", "LS", 0.999, false);
}

/** Loads reduced matrices Br and initializes the libROM matrix interpolator.
 *
 * One reduced matrix is read for each sampled parameter point. Identity
 * rotations are constructed, the closest sampled point to \p desired_point
 * is used as the reference index, and the interpolator is initialized for
 * subsequent online interpolation of Br.
 *
 * \param desired_point Online parameter point used to choose the reference sample.
 */
void
ROMProblem::SetupBrInterpolator(CAROM::Vector& desired_point)
{
  std::vector<std::shared_ptr<CAROM::Matrix>> Br_matrices;

  // Load Ar and rhs from libROM files
  for (size_t i = 0; i < param_points.size(); ++i)
  {
    const std::string Br_filename = "data/rom_system_Br_" + std::to_string(i);

    // Create empty containers
    auto Br = std::make_shared<CAROM::Matrix>();

    // Read matrix
    Br->local_read(Br_filename, opensn::mpi_comm.rank());
    Br_matrices.push_back(Br);
  }

  // Make Identity Rotations
  std::vector<std::shared_ptr<CAROM::Matrix>> rotations;
  int rom_dim = Br_matrices[0]->numRows();

  for (size_t i = 0; i < Br_matrices.size(); ++i)
  {
    std::shared_ptr<CAROM::Matrix> I;
    I = std::make_shared<CAROM::Matrix>(rom_dim, rom_dim, false);
    for (int j = 0; j < rom_dim; ++j)
      I->item(j, j) = 1.0;
    rotations.push_back(I);
  }

  int ref_index = getClosestPoint(param_points, desired_point);

  Br_interp_obj_ptr_ = std::make_unique<CAROM::MatrixInterpolator>(
    param_points, rotations, Br_matrices,
    ref_index, "R", "G", "LS", 0.999, false);
}

/** Interpolates Ar and rhs at a desired parameter point.
 *
 * Requires SetupArInterpolator() and SetupRHSrInterpolator() to have been called.
 *
 * \param desired_point Parameter at which to interpolate.
 * \param Ar_interp Output interpolated reduced matrix.
 * \param rhs_interp Output interpolated reduced RHS vector.
 */
void 
ROMProblem::InterpolateArAndRHSr(
    CAROM::Vector& desired_point,
    std::shared_ptr<CAROM::Matrix>& Ar_interp,
    std::shared_ptr<CAROM::Vector>& rhs_interp)
{
  Ar_interp = Ar_interp_obj_ptr_->interpolate(desired_point);
  rhs_interp = rhs_interp_obj_ptr_->interpolate(desired_point);
}

/** Interpolates Ar and Br at a desired parameter point.
 *
 * Requires SetupArInterpolator() and SetupBrInterpolator() to have been called.
 *
 * \param desired_point Parameter at which to interpolate.
 * \param Ar_interp Output interpolated reduced matrix.
 * \param Br_interp Output interpolated reduced production matrix.
 */
void 
ROMProblem::InterpolateArAndBr(
    CAROM::Vector& desired_point,
    std::shared_ptr<CAROM::Matrix>& Ar_interp,
    std::shared_ptr<CAROM::Matrix>& Br_interp)
{
  Ar_interp = Ar_interp_obj_ptr_->interpolate(desired_point);
  Br_interp = Br_interp_obj_ptr_->interpolate(desired_point);
}

/** Returns the schema for the nested `options` parameter block.
 *
 * The block controls ROM workflow phases and parameterization:
 * - phase: offline | merge | systems | online
 * - param_id: integer identifier for snapshot/system naming
 * - param_file: parameter matrix file path
 * - new_point: array specifying an online interpolation point
 */
InputParameters
ROMProblem::GetOptionsBlock()
{
  InputParameters params;

  params.SetGeneralDescription("Set options from a list of parameters");
  params.AddOptionalParameter("param_id", 0, "A parameter id for parametric problems.");
  params.AddOptionalParameter("phase", "offline", "The phase (offline, online, systems, or merge) for ROM purposes.");
  params.AddOptionalParameter("param_file", "", "A file containing an array of parameters for ROM.");
  params.AddOptionalParameterArray<double>("new_point", {0.0}, "New parameter point for ROM.");
  params.ConstrainParameterRange("phase", AllowableRangeList::New({"offline", "merge", "systems", "online"}));

  return params;
}

/** Parses and stores ROM options from an input block.
 *
 * Validates the provided block against GetOptionsBlock() and updates the internal
 * ROMOptions structure.
 *
 * \param input Parameter block containing ROM option values.
 */
void
ROMProblem::SetOptions(const InputParameters& input)
{
  auto params = ROMProblem::GetOptionsBlock();
  params.AssignParameters(input);

  for (size_t p = 0; p < params.GetNumParameters(); ++p)
  {
    const auto& spec = params.GetParam(p);

    if (spec.GetName() == "param_id")
      options_.param_id = spec.GetValue<int>();

    else if (spec.GetName() == "phase")
    {
      const std::map<std::string, Phase> phase_map =
      {
        {"offline", Phase::OFFLINE},
        {"merge",   Phase::MERGE},
        {"systems", Phase::SYSTEMS},
        {"online",  Phase::ONLINE}
      };
      const std::string phase_str = spec.GetValue<std::string>();
      auto it = phase_map.find(phase_str);
      options_.phase = it->second;
    }
    else if (spec.GetName() == "param_file")
      options_.param_file = spec.GetValue<std::string>();

    else if (spec.GetName() == "new_point")
    {
      spec.RequireBlockTypeIs(ParameterBlockType::ARRAY);

      std::vector<double> vals;
      vals.reserve(spec.GetNumParameters());
      for (const auto& sub_param : spec)
        vals.push_back(sub_param.GetValue<double>());

      options_.new_point = std::make_unique<CAROM::Vector>(static_cast<int>(vals.size()), false);
      for (int i = 0; i < static_cast<int>(vals.size()); ++i)
        (*(options_.new_point))(i) = vals[static_cast<size_t>(i)];
    }
  } // for p
}

ROMOptions&
ROMProblem::GetOptions()
{
  return options_;
}

} // namespace opensn
