// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "nl_keigen_rom_solver.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/compute/lbs_compute.h"
#include "framework/logging/log.h"
#include "framework/runtime.h"
#include "framework/utils/error.h"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <memory>

namespace opensn
{

InputParameters
NLKEigenROMSolver::GetInputParameters()
{
  InputParameters params = NonLinearKEigenSolver::GetInputParameters();

  params.SetGeneralDescription(
    "Implementation of a non-linear k-eigenvalue ROM solver. Offline phase runs the "
    "full-order non-linear k-eigen solver and takes samples with libROM; all reduced "
    "ROM phases follow the power-iteration ROM workflow.");
  params.ChangeExistingParamToOptional("name", "NLKEigenROMSolver");

  params.AddRequiredParameter<std::shared_ptr<Problem>>("rom_problem", "A ROM problem");

  return params;
}

NLKEigenROMSolver::NLKEigenROMSolver(const InputParameters& params)
  : Solver(params),
    lbs_problem_(params.GetSharedPtrParam<Problem, DiscreteOrdinatesProblem>("problem")),
    rom_problem_(params.GetSharedPtrParam<Problem, ROMProblem>("rom_problem")),
    nl_solver_(params)
{
}

void
NLKEigenROMSolver::Initialize()
{
  nl_solver_.Initialize();
  initialized_ = true;
}

void
NLKEigenROMSolver::Execute()
{
  OpenSnLogicalErrorIf(not initialized_, GetName() + ": Initialize must be called before Execute.");

  auto& rom_options = rom_problem_->GetOptions();

  if (rom_options.phase == Phase::OFFLINE)
  {
    auto start = std::chrono::high_resolution_clock::now();

    nl_solver_.Execute();
    k_eff_ = nl_solver_.GetEigenvalue();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    if (opensn::mpi_comm.rank() == 0)
    {
      std::ofstream outfile("results/offline_time_" + std::to_string(rom_options.param_id) +
                            ".txt");
      if (outfile.is_open())
        outfile << elapsed.count() << std::endl;
    }

    if (rom_options.take_sample)
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

    const std::string Ar_filename = "data/rom_system_Ar_" + std::to_string(rom_options.param_id);
    const std::string Br_filename = "data/rom_system_Br_" + std::to_string(rom_options.param_id);

    rom_problem_->AssembleROM(AU_, BU_, Ar_filename, Br_filename);
  }

  if (rom_options.phase == Phase::MIPOD)
  {
    rom_problem_->LoadUgs();

    auto start = std::chrono::high_resolution_clock::now();

    std::shared_ptr<CAROM::Matrix> AU_ = rom_problem_->AssembleAU();
    std::shared_ptr<CAROM::Matrix> BU_ = rom_problem_->AssembleBU();

    k_eff_ = rom_problem_->MIPOD(AU_, BU_);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    if (opensn::mpi_comm.rank() == 0)
    {
      std::ofstream outfile("results/mipod_time_" + std::to_string(rom_options.param_id) + ".txt");
      if (outfile.is_open())
        outfile << elapsed.count() << std::endl;
    }

    log.Log() << "\n";
    log.Log() << "        Final k-eigenvalue    :        " << std::setprecision(7) << k_eff_;
    log.Log() << "\n\n";

    log.Log() << "LinearBoltzmann::NLKEigenROMSolver MIPOD execution completed\n\n";
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
        outfile << elapsed.count() << std::endl;
    }

    log.Log() << "\n";
    log.Log() << "        Final k-eigenvalue    :        " << std::setprecision(7) << k_eff_;
    log.Log() << "\n\n";

    log.Log() << "LinearBoltzmann::NLKEigenROMSolver execution completed\n\n";
  }
}

double
NLKEigenROMSolver::GetEigenvalue() const
{
  return k_eff_;
}

} // namespace opensn
