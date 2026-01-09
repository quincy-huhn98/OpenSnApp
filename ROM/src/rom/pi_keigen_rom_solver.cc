// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/lbs_problem/compute/lbs_compute.h"
#include "framework/runtime.h"
#include "pi_keigen_rom_solver.h"
#include <chrono>
#include <fstream>
#include <memory>

namespace opensn
{

/** Returns the input-parameter schema for PowerIterationKEigenROMSolver.
 *
 * Extends the base power-iteration k-eigen solver schema with:
 * - `rom_problem` : an existing ROMProblem instance that manages ROM workflow.
 */
InputParameters
PowerIterationKEigenROMSolver::GetInputParameters()
{
  // Start from the base PI-k-eigen solver parameters
  InputParameters params = PowerIterationKEigenSolver::GetInputParameters();

  params.SetGeneralDescription(
    "Implementation of a k-eigenvalue ROM solver. Offline phase runs the "
    "full-order power-iteration k-eigen solver and takes sample with libROM.");
  params.ChangeExistingParamToOptional("name", "PowerIterationKEigenROMSolver");

  params.AddRequiredParameter<std::shared_ptr<Problem>>(
    "rom_problem", "A ROM problem");

  return params;
}

PowerIterationKEigenROMSolver::PowerIterationKEigenROMSolver(const InputParameters& params)
  : PowerIterationKEigenSolver(params),
    lbs_problem_(params.GetSharedPtrParam<Problem, DiscreteOrdinatesProblem>("problem")), 
    rom_problem_(params.GetSharedPtrParam<Problem, ROMProblem>("rom_problem"))
{
}

void
PowerIterationKEigenROMSolver::Initialize()
{
  PowerIterationKEigenSolver::Initialize();
}

/** Executes the requested ROM workflow phase for the k-eigenvalue solver.
 *
 * Supported phases are:
 * - OFFLINE : runs the full-order power iteration and writes snapshots,
 * - MERGE   : builds the reduced bases from stored snapshots,
 * - SYSTEMS : assembles and writes reduced operators,
 * - ONLINE  : interpolates and solves the reduced k-eigenvalue system.
 */
void
PowerIterationKEigenROMSolver::Execute()
{
  auto& rom_options = rom_problem_->GetOptions();
  auto& options = lbs_problem_->GetOptions();

  if (rom_options.phase == Phase::OFFLINE)
  {
    // Run full-order k-eigen solve and time it
    auto start = std::chrono::high_resolution_clock::now();

    PowerIterationKEigenSolver::Execute();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    if (opensn::mpi_comm.rank() == 0)
    {
      std::ofstream outfile("results/offline_time_" + std::to_string(rom_options.param_id) + ".txt");
      if (outfile.is_open())
      {
        outfile << elapsed.count() << std::endl;
        outfile.close();
      }
    }

    rom_problem_->TakeSample(rom_options.param_id);
  }
  if (rom_options.phase == Phase::MERGE)
  {
    rom_problem_->MergePhase(rom_options.param_id);
  }
  if (rom_options.phase == Phase::SYSTEMS)
  {
    std::shared_ptr<CAROM::Matrix> AU_ = rom_problem_->AssembleAU();
    rom_problem_->LoadUgs();
    std::shared_ptr<CAROM::Matrix> BU_ = rom_problem_->AssembleBU();

    const std::string Ar_filename  =
      "data/rom_system_Ar_" + std::to_string(rom_options.param_id);
    const std::string Br_filename =
      "data/rom_system_Br_" + std::to_string(rom_options.param_id);

    rom_problem_->AssembleROM(AU_, BU_, Ar_filename, Br_filename);
  }
  if (rom_options.phase == Phase::ONLINE)
  {
    rom_problem_->ReadParamMatrix(rom_options.param_file);
    rom_problem_->LoadUgs();

    std::shared_ptr<CAROM::Matrix> Ar_interp;
    std::shared_ptr<CAROM::Matrix> Br_interp;

    rom_problem_->SetupArInterpolator(*rom_options.new_point);
    rom_problem_->SetupBrInterpolator(*rom_options.new_point);

    auto start = std::chrono::high_resolution_clock::now();

    rom_problem_->InterpolateArAndBr(*rom_options.new_point, Ar_interp, Br_interp);

    k_eff_ = rom_problem_->SolveROM(Ar_interp, Br_interp);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    if (opensn::mpi_comm.rank() == 0)
    {
      std::ofstream outfile("results/online_time_" + std::to_string(rom_options.param_id) + ".txt");
      if (outfile.is_open())
      {
        outfile << elapsed.count() << std::endl;
        outfile.close();
      }
    }

    log.Log() << "\n";
    log.Log() << "        Final k-eigenvalue    :        " << std::setprecision(7) << k_eff_;
    log.Log() << "\n\n";

    log.Log() << "LinearBoltzmann::KEigenvalueROMSolver execution completed\n\n";
  }
}

} // namespace opensn
