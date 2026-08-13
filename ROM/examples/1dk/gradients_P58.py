#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P58 k-eigenvalue gradient deck for active-subspace offline data.

This deck accepts the same global inputs used by base_P58.py.  For an
``offline`` run, it solves the forward k-eigenvalue problem, solves the
corresponding adjoint problem, evaluates dk/dp for the eight P58 parameters,
and writes the gradient row to ``data/gradients_<pid>.txt``.
"""

import os
import numpy as np


def _get_rank():
    try:
        return mpi_comm.rank
    except NameError:
        try:
            from mpi4py import MPI

            return MPI.COMM_WORLD.rank
        except Exception:
            return 0


def _scalar_pp_value(pp):
    value = pp.GetValue()
    return float(value[0][0])


def _keigen_scaled_sensitivity(problem, k_eff, pp_kwargs):
    pp = CrossSectionSensitivityPostprocessor(problem=problem, **pp_kwargs)
    pp.Execute()
    pp.ApplyKEigenvalueScaling(k_eff)
    return _scalar_pp_value(pp)


if __name__ == "__main__":
    rank = _get_rank()

    try:
        param_id = int(pid)
    except NameError:
        param_id = 0

    try:
        run_phase = phase
    except NameError:
        run_phase = "offline"

    try:
        write_h5 = bool(saveh5)
    except NameError:
        write_h5 = False

    if rank == 0:
        print("Parameter id = {}".format(param_id))
        print("{} phase".format(run_phase))

    widths = [4.6, 1.126152]
    nrefs = [500, 500]
    nodes = [0.0]
    for imat in range(len(widths)):
        dx = widths[imat] / nrefs[imat]
        for _ in range(nrefs[imat]):
            nodes.append(nodes[-1] + dx)

    meshgen = OrthogonalMeshGenerator(node_sets=[nodes])
    grid = meshgen.Execute()
    grid.SetUniformBlockID(0)

    # Fissile slab: this is the only block whose cross sections are parameterized.
    lv = RPPLogicalVolume(infx=True, infy=True, zmin=0.0, zmax=4.6)
    grid.SetBlockIDFromLogicalVolume(lv, 1, True)

    num_groups = 2

    scatt = MultiGroupXS()
    scatt.LoadFromOpenSn("data/H2O_mg.xs")

    fissile = MultiGroupXS()
    fissile.LoadFromOpenSn("data/URRb.xs")

    n_angles = 128
    scat_order = 0
    pquad = GLProductQuadrature1DSlab(n_polar=n_angles, scattering_order=scat_order)

    boundary_conditions = [{"name": "zmin", "type": "reflecting"}]

    phys = DiscreteOrdinatesProblem(
        mesh=grid,
        num_groups=num_groups,
        groupsets=[
            {
                "groups_from_to": (0, num_groups - 1),
                "angular_quadrature": pquad,
                "inner_linear_method": "petsc_gmres",
                "l_max_its": 50,
                "gmres_restart_interval": 50,
                "l_abs_tol": 1.0e-10,
            },
        ],
        xs_map=[
            {"block_ids": [0], "xs": scatt},
            {"block_ids": [1], "xs": fissile},
        ],
        boundary_conditions=boundary_conditions,
        options={
            "use_precursors": False,
            "save_angular_flux": True,
            "verbose_inner_iterations": False,
            "verbose_outer_iterations": True,
        },
    )

    rom_options = {"param_id": param_id, "phase": run_phase}

    rom = ROMProblem(problem=phys, options=rom_options)

    k_solver = PowerIterationROMSolver(problem=phys, rom_problem=rom, k_tol=1.0e-7)
    k_solver.Initialize()
    k_solver.Execute()
    k_eff = k_solver.GetEigenvalue()

    if run_phase == "online":
        if write_h5:
            phys.WriteFluxMoments("output/rom_{}_".format(param_id))
            if rank == 0:
                np.savetxt("output/rom_k_{}.txt".format(param_id), [k_eff])
    elif run_phase == "mipod":
        if write_h5:
            phys.WriteFluxMoments("output/mipod_{}_".format(param_id))
            if rank == 0:
                np.savetxt("output/mipod_k_{}.txt".format(param_id), [k_eff])
    else:
        if write_h5:
            phys.WriteFluxMoments("output/fom_{}_".format(param_id))
            if rank == 0:
                np.savetxt("output/fom_k_{}.txt".format(param_id), [k_eff])

    # Only the offline FOM samples are used to build the active-subspace
    # gradient matrix.
    if run_phase == "offline":
        fwd_phi_prefix = "output/grad_p58_fwd_phi_{}_".format(param_id)
        adj_phi_prefix = "output/grad_p58_adj_phi_{}_".format(param_id)
        fwd_psi_prefix = "output/grad_p58_fwd_psi_{}_".format(param_id)
        adj_psi_prefix = "output/grad_p58_adj_psi_{}_".format(param_id)

        phys.WriteFluxMoments(fwd_phi_prefix)
        phys.WriteAngularFluxes(fwd_psi_prefix)

        phys.SetAdjoint(True)
        rom_options = {"param_id": param_id, "phase": run_phase, "take_sample": False}
        rom = ROMProblem(problem=phys, options=rom_options)

        phys.SetBoundaryOptions(boundary_conditions=boundary_conditions)
        adj_k_solver = PowerIterationROMSolver(
            problem=phys,
            rom_problem=rom,
            k_tol=1.0e-7,
        )
        adj_k_solver.Initialize()
        adj_k_solver.Execute()
        phys.WriteFluxMoments(adj_phi_prefix)
        phys.WriteAngularFluxes(adj_psi_prefix)

        common_phi = {
            "forward_flux_moments": fwd_phi_prefix,
            "adjoint_flux_moments": adj_phi_prefix,
            "block_ids": [1],
        }
        common_psi = {
            "forward_angular_fluxes": fwd_psi_prefix,
            "adjoint_angular_fluxes": adj_psi_prefix,
            "block_ids": [1],
        }

        gradient = np.zeros(6)

        # Parameters 0-1: sigma_f[g].  The postprocessor's production
        # sensitivity gives dk/dsigma_f[g] because it internally multiplies
        # by nu = nu_sigma_f / sigma_f when relative=False.
        for g in range(num_groups):
            gradient[g] = _keigen_scaled_sensitivity(
                phys,
                k_eff,
                {
                    "sensitivity_type": "production",
                    "group": g,
                    **common_phi,
                },
            )

        # Parameters 4-7: zeroth-moment transfer entries S[from_group,to_group]
        # in the same order used by P58Problem.bounds and URRb_base.txt.
        scatter_pairs = [(0, 0), (0, 1), (1, 0), (1, 1)]
        for j, (from_group, to_group) in enumerate(scatter_pairs):
            gradient[2 + j] = _keigen_scaled_sensitivity(
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

        if rank == 0:
            os.makedirs("data", exist_ok=True)
            os.makedirs("output", exist_ok=True)
            np.savetxt("data/gradients_{}.txt".format(param_id), gradient[None, :])
            np.savetxt("output/fom_k_{}.txt".format(pid), [k_solver.GetEigenvalue()])