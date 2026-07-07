#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C5G7 k-eigenvalue gradient deck using the nonlinear ROM eigensolver.

The forward and adjoint solves use the same nonlinear solver inputs as the
reference C5G7 nonlinear solve, with the ROM wrapper added.  The gradient order
matches xs.CrossSections for the parameterized fuel materials:

    UO2, 7pMOX, 4_3pMOX, 8_7pMOX,

and within each material:

    [sigma_f[g], sigma_c[g], S_ell0[g_from,g_to] for entries in the XS file].
"""

import os
import sys
import numpy as np

if "opensn_console" not in globals():
    from mpi4py import MPI

    rank = MPI.COMM_WORLD.rank
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))
    from pyopensn.aquad import GLCProductQuadrature2DXY
    from pyopensn.mesh import FromFileMeshGenerator, PETScGraphPartitioner
    from pyopensn.xs import MultiGroupXS
    from pyopensn.solver import DiscreteOrdinatesProblem
    from pyopensn.post import CrossSectionSensitivityPostprocessor
else:
    try:
        rank = mpi_comm.rank
    except NameError:
        rank = 0


def _global_int(name, default):
    try:
        return int(globals()[name])
    except KeyError:
        return default


def _global_bool(name, default):
    try:
        return bool(globals()[name])
    except KeyError:
        return default


def _global_string(name, default):
    try:
        return str(globals()[name])
    except KeyError:
        return default


def _scalar_pp_value(pp):
    value = pp.GetValue()
    return float(value[0][0])


def _keigen_scaled_sensitivity(problem, k_eff, pp_kwargs):
    pp = CrossSectionSensitivityPostprocessor(problem=problem, **pp_kwargs)
    pp.Execute()
    pp.ApplyKEigenvalueScaling(k_eff)
    return _scalar_pp_value(pp)


def _read_transfer_entries(xs_file):
    entries = []
    in_block = False
    with open(xs_file, "r") as f:
        for line in f:
            stripped = line.strip()
            if stripped == "TRANSFER_MOMENTS_BEGIN":
                in_block = True
                continue
            if stripped == "TRANSFER_MOMENTS_END":
                break
            if not in_block:
                continue

            toks = stripped.split()
            if len(toks) < 5:
                continue
            if toks[0] not in ("M_GFROM_GTO_VAL", "M_GPRIME_G_VAL"):
                continue
            ell = int(toks[1])
            if ell != 0:
                continue
            entries.append((int(toks[2]), int(toks[3])))
    return entries


if __name__ == "__main__":
    param_id = _global_int("pid", 0)
    run_phase = _global_string("phase", "offline")
    write_h5 = _global_bool("saveh5", False)

    if rank == 0:
        print("Parameter id = {}".format(param_id))
        print("{} phase".format(run_phase))
        print("Running C5G7 gradients with nonlinear ROM eigensolver")

    meshgen = FromFileMeshGenerator(
        filename="mesh/lattice_C5G7_3x3.obj",
        partitioner=PETScGraphPartitioner(type="parmetis"),
    )
    grid = meshgen.Execute()
    grid.SetOrthogonalBoundaries()

    xss = [MultiGroupXS() for _ in range(7)]
    xss[6].LoadFromOpenSn("data/XS_water.xs")
    xss[5].LoadFromOpenSn("data/XS_UO2.xs")
    xss[3].LoadFromOpenSn("data/XS_7pMOX.xs")
    xss[1].LoadFromOpenSn("data/XS_guide_tube.xs")
    xss[4].LoadFromOpenSn("data/XS_4_3pMOX.xs")
    xss[2].LoadFromOpenSn("data/XS_8_7pMOX.xs")
    xss[0].LoadFromOpenSn("data/XS_fission_chamber.xs")

    num_groups = xss[0].num_groups
    xs_map = [{"block_ids": [m], "xs": xss[m]} for m in range(len(xss))]

    pquad = GLCProductQuadrature2DXY(n_polar=8, n_azimuthal=32, scattering_order=0)
    boundary_conditions = [
        {"name": "xmin", "type": "reflecting"},
        {"name": "ymax", "type": "reflecting"},
    ]

    phys = DiscreteOrdinatesProblem(
        mesh=grid,
        num_groups=num_groups,
        groupsets=[
            {
                "groups_from_to": (0, num_groups - 1),
                "angular_quadrature": pquad,
                "inner_linear_method": "petsc_gmres",
                "angle_aggregation_type": "polar",
                "angle_aggregation_num_subsets": 1,
            },
        ],
        xs_map=xs_map,
        boundary_conditions=boundary_conditions,
        options={
            "verbose_outer_iterations": True,
            "verbose_inner_iterations": True,
            "save_angular_flux": True,
            "power_default_kappa": 1.0,
        },
        sweep_type="AAH",
    )

    rom = ROMProblem(problem=phys, options={"param_id": param_id, "phase": run_phase})

    k_solver = NLKEigenROMSolver(
        problem=phys,
        rom_problem=rom,
        nl_max_its=10,
        l_max_its=50,
        l_abs_tol=1.0e-8,
        l_rel_tol=1.0e-8,
    )

    k_solver.Initialize()
    k_solver.Execute()
    k_eff = k_solver.GetEigenvalue()

    if write_h5 and run_phase == "offline":
        phys.WriteFluxMoments("output/fom_{}_".format(param_id))
        if rank == 0:
            np.savetxt("output/fom_k_{}.txt".format(param_id), [k_eff])

    if run_phase == "offline":
        fwd_phi_prefix = "output/grad_c5g7_fwd_phi_{}_".format(param_id)
        adj_phi_prefix = "output/grad_c5g7_adj_phi_{}_".format(param_id)
        fwd_psi_prefix = "output/grad_c5g7_fwd_psi_{}_".format(param_id)
        adj_psi_prefix = "output/grad_c5g7_adj_psi_{}_".format(param_id)

        phys.WriteFluxMoments(fwd_phi_prefix)
        phys.WriteAngularFluxes(fwd_psi_prefix)

        phys.SetAdjoint(True)
        rom = ROMProblem(
            problem=phys,
            options={"param_id": param_id, "phase": run_phase, "take_sample": False},
        )
        phys.SetBoundaryOptions(boundary_conditions=boundary_conditions)

        adj_k_solver = NLKEigenROMSolver(
            problem=phys,
            rom_problem=rom,
            nl_max_its=10,
            l_max_its=50,
            l_abs_tol=1.0e-8,
            l_rel_tol=1.0e-8,
        )
        adj_k_solver.Initialize()
        adj_k_solver.Execute()
        phys.WriteFluxMoments(adj_phi_prefix)
        phys.WriteAngularFluxes(adj_psi_prefix)

        fuel_materials = [
            {"name": "UO2", "block_id": 5, "xs_reference": "materials/XS_UO2.xs"},
            {"name": "7pMOX", "block_id": 3, "xs_reference": "materials/XS_7pMOX.xs"},
            {"name": "4_3pMOX", "block_id": 4, "xs_reference": "materials/XS_4_3pMOX.xs"},
            {"name": "8_7pMOX", "block_id": 2, "xs_reference": "materials/XS_8_7pMOX.xs"},
        ]

        gradient_entries = []
        for material in fuel_materials:
            block_id = material["block_id"]
            transfer_entries = _read_transfer_entries(material["xs_reference"])

            common_phi = {
                "forward_flux_moments": fwd_phi_prefix,
                "adjoint_flux_moments": adj_phi_prefix,
                "block_ids": [block_id],
            }
            common_psi = {
                "forward_angular_fluxes": fwd_psi_prefix,
                "adjoint_angular_fluxes": adj_psi_prefix,
                "block_ids": [block_id],
            }

            # sigma_f[g]
            for g in range(num_groups):
                gradient_entries.append(
                    _keigen_scaled_sensitivity(
                        phys,
                        k_eff,
                        {
                            "sensitivity_type": "production",
                            "group": g,
                            **common_phi,
                        },
                    )
                )

            # sigma_c[g]. In this XS parameterization, capture changes sigma_t
            # without a compensating production or scattering term.
            for g in range(num_groups):
                gradient_entries.append(
                    _keigen_scaled_sensitivity(
                        phys,
                        k_eff,
                        {
                            "sensitivity_type": "sigma_t",
                            "group": g,
                            **common_psi,
                        },
                    )
                )

            # S_0[g_from,g_to]
            for from_group, to_group in transfer_entries:
                gradient_entries.append(
                    _keigen_scaled_sensitivity(
                        phys,
                        k_eff,
                        {
                            "sensitivity_type": "scatter",
                            "moment": 0,
                            "from_group": from_group,
                            "to_group": to_group,
                            **common_phi,
                        },
                    )
                )

        gradient = np.asarray(gradient_entries, dtype=float)

        if rank == 0:
            os.makedirs("data", exist_ok=True)
            os.makedirs("output", exist_ok=True)
            np.savetxt("data/gradients_{}.txt".format(param_id), gradient[None, :])
            np.savetxt("output/fom_k_{}.txt".format(param_id), [k_eff])
