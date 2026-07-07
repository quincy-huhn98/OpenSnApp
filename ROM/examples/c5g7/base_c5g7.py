#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C5G7 k-eigenvalue ROM deck using the nonlinear ROM eigensolver.

The mesh, material block mapping, quadrature, boundary conditions, sweep type,
and nonlinear solver tolerances match the reference C5G7 nonlinear deck.  The
phase/parameter handling follows the current ROM driver convention.
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


def _collect_parameter_vector():
    """Collect p0, p1, ... from OpenSn -p arguments."""
    values = []
    i = 0
    while True:
        name = "p{}".format(i)
        if name not in globals():
            break
        values.append(float(globals()[name]))
        i += 1
    return values



if __name__ == "__main__":
    param_id = _global_int("pid", 0)
    run_phase = _global_string("phase", "offline")
    write_h5 = _global_bool("saveh5", False)

    if rank == 0:
        print("Parameter id = {}".format(param_id))
        print("{} phase".format(run_phase))
        print("Running C5G7 with nonlinear ROM eigensolver")

    mesh_file = "mesh/lattice_C5G7_3x3.obj"
    meshgen = FromFileMeshGenerator(
        filename=mesh_file,
        partitioner=PETScGraphPartitioner(type="parmetis"),
    )
    grid = meshgen.Execute()
    grid.SetOrthogonalBoundaries()

    # Material/block ordering matches the reference nonlinear C5G7 deck.
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
        boundary_conditions=[
            {"name": "xmin", "type": "reflecting"},
            {"name": "ymax", "type": "reflecting"},
        ],
        options={
            "verbose_outer_iterations": True,
            "verbose_inner_iterations": True,
            "save_angular_flux": False,
            "power_default_kappa": 1.0,
        },
        sweep_type="CBC",
    )

    if run_phase == "online":
        try:
            point_file = pfile
            new_point = np.loadtxt(point_file).ravel().tolist()
        except NameError:
            new_point = _collect_parameter_vector()

        rom_options = {
            "param_id": param_id,
            "phase": run_phase,
            "param_file": "data/params_AS.txt" if os.path.exists("data/params_AS.txt") else "data/params.txt",
            "new_point": new_point,
        }
    else:
        rom_options = {"param_id": param_id, "phase": run_phase}

    rom = ROMProblem(problem=phys, options=rom_options)

    if run_phase == "offline":
        k_solver = NLKEigenROMSolver(
            problem=phys,
            rom_problem=rom,
            nl_max_its=10,
            l_max_its=50,
            l_abs_tol=1.0e-8,
            l_rel_tol=1.0e-8,
        )
    else:
        k_solver = PowerIterationROMSolver(problem=phys, rom_problem=rom, k_tol=1.0e-7)

    k_solver.Initialize()
    k_solver.Execute()
    k_eff = k_solver.GetEigenvalue()

    if write_h5:
        if run_phase == "online":
            phys.WriteFluxMoments("output/rom_{}_".format(param_id))
            if rank == 0:
                np.savetxt("output/rom_k_{}.txt".format(param_id), [k_eff])
        elif run_phase == "mipod":
            phys.WriteFluxMoments("output/mipod_{}_".format(param_id))
            if rank == 0:
                np.savetxt("output/mipod_k_{}.txt".format(param_id), [k_eff])
        elif run_phase == "offline":
            phys.WriteFluxMoments("output/fom_{}_".format(param_id))
            if rank == 0:
                np.savetxt("output/fom_k_{}.txt".format(param_id), [k_eff])
    elif run_phase == "offline":    
        if rank == 0:
            np.savetxt("output/fom_k_{}.txt".format(param_id), [k_eff])
