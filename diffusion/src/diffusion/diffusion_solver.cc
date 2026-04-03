// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "diffusion_solver.h"
#include "opensn/framework/field_functions/field_function_grid_based.h"
#include "opensn/framework/mesh/mesh_continuum/mesh_continuum.h"
#include "opensn/framework/runtime.h"

namespace opensn
{

DiffusionSolverBase::Boundary::Boundary() : type(BoundaryType::Dirichlet), values({0.0, 0.0, 0.0})
{
}

DiffusionSolverBase::Boundary::Boundary(BoundaryType type, const std::array<double, 3>& values)
  : type(type), values(values)
{
}

//

InputParameters
DiffusionSolverBase::GetInputParameters()
{
  InputParameters params = Solver::GetInputParameters();
  params.AddRequiredParameter<std::shared_ptr<MeshContinuum>>("mesh", "Mesh");
  params.AddOptionalParameter<double>("residual_tolerance", 1.0e-2, "Solver relative tolerance");
  params.AddOptionalParameter<int>("max_iters", 500, "Solver relative tolerance");
  return params;
}

InputParameters
DiffusionSolverBase::GetOptionsBlock()
{
  InputParameters params;
  params.AddOptionalParameterArray(
    "boundary_conditions", {}, "An array contain tables for each boundary specification.");
  params.LinkParameterToBlock("boundary_conditions", "DiffusionSolver::BoundaryOptionsBlock");
  return params;
}

InputParameters
DiffusionSolverBase::GetBoundaryOptionsBlock()
{
  InputParameters params;
  params.SetGeneralDescription("Set options for boundary conditions");
  params.AddRequiredParameter<std::string>("boundary",
                                           "Boundary to apply the boundary condition to.");
  params.AddRequiredParameter<std::string>("type", "Boundary type specification.");
  params.AddOptionalParameterArray<double>("coeffs", {}, "Coefficients.");
  return params;
}

DiffusionSolverBase::DiffusionSolverBase(const std::string& name,
                                         std::shared_ptr<MeshContinuum> grid_ptr)
  : Solver(name),
    grid_ptr_(std::move(grid_ptr)),
    num_local_dofs_(0),
    num_global_dofs_(0),
    x_(nullptr),
    b_(nullptr),
    A_(nullptr),
    residual_tolerance_(1.0e-2),
    max_iters_(500)
{
}

DiffusionSolverBase::DiffusionSolverBase(const InputParameters& params)
  : Solver(params),
    grid_ptr_(params.GetParamValue<std::shared_ptr<MeshContinuum>>("mesh")),
    num_local_dofs_(0),
    num_global_dofs_(0),
    x_(nullptr),
    b_(nullptr),
    A_(nullptr),
    residual_tolerance_(params.GetParamValue<double>("residual_tolerance")),
    max_iters_(params.GetParamValue<int>("max_iters"))
{
}

DiffusionSolverBase::~DiffusionSolverBase()
{
  VecDestroy(&x_);
  VecDestroy(&b_);
  MatDestroy(&A_);
}

void
DiffusionSolverBase::SetOptions(const InputParameters& params)
{
  for (size_t p = 0; p < params.GetNumParameters(); ++p)
  {
    const auto& spec = params.GetParam(p);
    if (spec.GetName() == "boundary_conditions")
    {
      spec.RequireBlockTypeIs(ParameterBlockType::ARRAY);
      for (size_t b = 0; b < spec.GetNumParameters(); ++b)
      {
        auto bndry_params = GetBoundaryOptionsBlock();
        bndry_params.AssignParameters(spec.GetParam(b));
        SetBoundaryOptions(bndry_params);
      }
    }
  }
}

void
DiffusionSolverBase::InitFieldFunctions()
{
  if (field_functions_.empty())
  {
    std::string solver_name;
    if (not GetName().empty())
      solver_name = GetName() + "-";

    std::string name = solver_name + "phi";

    auto initial_field_function =
      std::make_shared<FieldFunctionGridBased>(name, sdm_ptr_, Unknown(UnknownType::SCALAR));

    field_functions_.push_back(initial_field_function);
    field_function_stack.push_back(initial_field_function);
  } // if not ff set
}

void
DiffusionSolverBase::ValidateMaterialFunctions() const
{
  if (!d_coef_function_)
    throw std::logic_error("DiffusionSolverBase: D coefficient function not set.");
  if (!q_ext_function_)
    throw std::logic_error("DiffusionSolverBase: Q_ext function not set.");
  if (!sigma_a_function_)
    throw std::logic_error("DiffusionSolverBase: Sigma_a function not set.");
}

void
DiffusionSolverBase::UpdateFieldFunctions()
{
  auto& ff = *field_functions_.front();
  ff.UpdateFieldVector(x_);
}

const std::vector<std::shared_ptr<FieldFunctionGridBased>>&
DiffusionSolverBase::GetFieldFunctions() const
{
  return field_functions_;
}

} // namespace opensn
