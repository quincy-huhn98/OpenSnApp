// SPDX-FileCopyrightText: 2025 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/solvers/steady_state_solver.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/lbs_problem.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/iterative_methods/ags_linear_solver.h"
#include "framework/runtime.h"
#include "steady_state_rom_solver.h"
#include <memory>
#include <fstream>
#include <chrono>

namespace opensn
{

InputParameters
SteadyStateROMSolver::GetInputParameters()
{
  InputParameters params = Solver::GetInputParameters();

  params.SetGeneralDescription("Implementation of a steady state ROM solver. This solver calls the "
                               "across-groupset (AGS) solver offline and interfaces with libROM.");
  params.ChangeExistingParamToOptional("name", "SteadyStateROMSolver");
  params.AddRequiredParameter<std::shared_ptr<Problem>>("problem", "An existing lbs problem");
  params.AddRequiredParameter<std::shared_ptr<Problem>>("rom_problem", "A ROM problem");

  return params;
}

SteadyStateROMSolver::SteadyStateROMSolver(const InputParameters& params)
  : SteadyStateSourceSolver(params), 
  lbs_problem_(params.GetSharedPtrParam<Problem, LBSProblem>("problem")), 
  rom_problem_(params.GetSharedPtrParam<Problem, ROMProblem>("rom_problem"))
{
}

void
SteadyStateROMSolver::Initialize()
{
  lbs_problem_->Initialize();
}

void
SteadyStateROMSolver::Execute()
{
  auto& options = lbs_problem_->GetOptions();
  auto& rom_options = rom_problem_->GetOptions();

  if (rom_options.phase == "offline")
  {
    auto& ags_solver = *lbs_problem_->GetAGSSolver();

    auto start = std::chrono::high_resolution_clock::now();
    ags_solver.Solve();
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed = end - start;
    if (opensn::mpi_comm.rank() == 0)
    {
      std::ofstream outfile("results/offline_time.txt");
      if (outfile.is_open()) {
        outfile << elapsed.count() <<std::endl;
        outfile.close();
      }
    }

    lbs_problem_->UpdateFieldFunctions();

    rom_problem_->TakeSample(rom_options.param_id);
  }
  if (rom_options.phase == "merge")
  {
    rom_problem_->MergePhase(rom_options.param_id);
  }
  if (rom_options.phase == "systems")
  {
    rom_problem_->LoadUgs();

    std::shared_ptr<CAROM::Matrix> AU_ = rom_problem_->AssembleAU();
    std::shared_ptr<CAROM::Vector> b_ = rom_problem_->AssembleRHS();
    const std::string& Ar_filename = "data/rom_system_Ar_" + std::to_string(rom_options.param_id);
    const std::string& rhs_filename = "data/rom_system_br_" + std::to_string(rom_options.param_id);
    rom_problem_->AssembleROM(AU_, b_, Ar_filename, rhs_filename);
  }
  if (rom_options.phase == "online")
  {
    rom_problem_->ReadParamMatrix(rom_options.param_file);
    rom_problem_->LoadUgs();

    std::shared_ptr<CAROM::Matrix> Ar_interp;
    std::shared_ptr<CAROM::Vector> rhs_interp;
    rom_problem_->SetupArInterpolator(*rom_options.new_point);
    rom_problem_->SetuprhsInterpolator(*rom_options.new_point);

    auto start = std::chrono::high_resolution_clock::now();

    rom_problem_->InterpolateArAndRHS(*rom_options.new_point, Ar_interp, rhs_interp);
    rom_problem_->SolveROM(Ar_interp, rhs_interp);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    if (opensn::mpi_comm.rank() == 0)
    {
      std::ofstream outfile("results/online_time.txt");
      if (outfile.is_open()) 
      {
        outfile << elapsed.count() <<std::endl;
        outfile.close();
      }
    }
  }
}

} // namespace opensn
