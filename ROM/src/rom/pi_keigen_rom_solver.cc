// SPDX-FileCopyrightText: 2025 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "pi_keigen_rom_solver.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/lbs_compute.h"
#include "framework/runtime.h"

#include <chrono>
#include <fstream>
#include <memory>

namespace opensn
{

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
    lbs_problem_(params.GetSharedPtrParam<Problem, LBSProblem>("problem")), 
    rom_problem_(params.GetSharedPtrParam<Problem, ROMProblem>("rom_problem"))
{
}

void
PowerIterationKEigenROMSolver::Initialize()
{
  PowerIterationKEigenSolver::Initialize();
}

void
PowerIterationKEigenROMSolver::Execute()
{
  auto& rom_options = rom_problem_->GetOptions();
  auto& options = lbs_problem_->GetOptions();

  if (rom_options.phase == "offline")
  {
    // Run full-order k-eigen solve and time it
    auto start = std::chrono::high_resolution_clock::now();

    PowerIterationKEigenSolver::Execute();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    if (opensn::mpi_comm.rank() == 0)
    {
      std::ofstream outfile("results/offline_time.txt");
      if (outfile.is_open())
      {
        outfile << elapsed.count() << std::endl;
        outfile.close();
      }
    }

    rom_problem_->TakeSample(rom_options.param_id);
  }
  else if (rom_options.phase == "merge")
  {
    rom_problem_->MergePhase(rom_options.param_id);
  }
  else if (rom_options.phase == "systems")
  {
    rom_problem_->LoadUgs();
    std::shared_ptr<CAROM::Matrix> AU_ = rom_problem_->AssembleAU();
    std::shared_ptr<CAROM::Matrix> BU_ = rom_problem_->AssembleBU();

    const std::string Ar_filename  =
      "data/rom_system_Ar_" + std::to_string(rom_options.param_id);
    const std::string Br_filename =
      "data/rom_system_Br_" + std::to_string(rom_options.param_id);

    rom_problem_->AssembleROM(AU_, BU_, Ar_filename, Br_filename);
  }
  else if (rom_options.phase == "online")
  {
    rom_problem_->ReadParamMatrix(rom_options.param_file);
    rom_problem_->LoadUgs();

    std::shared_ptr<CAROM::Matrix> Ar_interp;
    std::shared_ptr<CAROM::Matrix> Br_interp;

    rom_problem_->SetupArInterpolator(*rom_options.new_point);
    rom_problem_->SetupBrInterpolator(*rom_options.new_point);

    auto start = std::chrono::high_resolution_clock::now();

    rom_problem_->InterpolateArAndBr(*rom_options.new_point, Ar_interp, Br_interp);

    rom_problem_->InitializeSolver(Ar_interp, Br_interp);

    auto& options = lbs_problem_->GetOptions();
    double k_eff_prev = 1.0;
    double k_eff_change = 1.0;

    // Start power iterations
    size_t nit = 0;
    bool converged = false;
    while (nit < max_iters_)
    {
      // This solves the inners for transport
      rom_problem_->SolveROM(k_eff_);
      const auto F_new = ComputeFissionProduction(*lbs_problem_, phi_new_local_);
      k_eff_ = F_new / F_prev_ * k_eff_;
      F_prev_ = F_new;

      const double reactivity = (k_eff_ - 1.0) / k_eff_;

      // Check convergence, bookkeeping
      k_eff_change = fabs(k_eff_ - k_eff_prev) / k_eff_;
      k_eff_prev = k_eff_;
      nit += 1;

      converged = k_eff_change < std::max(k_tolerance_, 1.0e-12);

      // Print iteration summary
      if (options.verbose_outer_iterations)
      {
        std::stringstream k_iter_info;
        k_iter_info << "  Iteration " << std::setw(5) << nit << "  k_eff " << std::setw(11)
                    << std::setprecision(7) << k_eff_ << "  k_eff change " << std::setw(12)
                    << k_eff_change << "  reactivity " << std::setw(10) << reactivity * 1e5;
        if (converged)
          k_iter_info << " CONVERGED\n";

        log.Log() << k_iter_info.str();
      }

      if (converged)
        break;
    } // for k iterations

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    if (opensn::mpi_comm.rank() == 0)
    {
      std::ofstream outfile("results/online_time.txt");
      if (outfile.is_open())
      {
        outfile << elapsed.count() << std::endl;
        outfile.close();
      }
    }

    log.Log() << "\n";
    log.Log() << "        Final k-eigenvalue    :        " << std::setprecision(7) << k_eff_;
    log.Log() << "        Final change          :        " << std::setprecision(6) << k_eff_change
              << "\n\n";

    lbs_problem_->UpdateFieldFunctions();

    log.Log() << "LinearBoltzmann::KEigenvalueROMSolver execution completed\n\n";
  }
}

} // namespace opensn
