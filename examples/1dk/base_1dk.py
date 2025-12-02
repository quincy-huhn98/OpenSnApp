#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Standard Reed 1D 1-group problem

import os
import sys

if "opensn_console" not in globals():
    from mpi4py import MPI
    size = MPI.COMM_WORLD.size
    rank = MPI.COMM_WORLD.rank
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))
    from pyopensn.mesh import OrthogonalMeshGenerator
    from pyopensn.xs import MultiGroupXS
    from pyopensn.source import VolumetricSource
    from pyopensn.aquad import GLProductQuadrature1DSlab
    from pyopensn.solver import DiscreteOrdinatesProblem, SteadyStateSolver
    from pyopensn.logvol import RPPLogicalVolume
    from pyopensn.rom import ROMProblem, SteadyStateROMSolver

if __name__ == "__main__":

    try:
        print("Nu Parameter = {}".format(nu))
        param = nu
    except:
        nu=1.99
        print("Nu Nominal = {}".format(nu))

    try:
        print("Cross Section Parameter = {}".format(sigma_f))
        param = sigma_f
    except:
        sigma_f=0.35
        print("Cross Section Nominal = {}".format(sigma_f))

    try:
        print("Parameter id = {}".format(p_id))
    except:
        p_id=0
        print("Parameter id = {}".format(p_id))

    try:
        if phase == 0:
            print("Offline Phase")
            phase = "offline"
        elif phase == 1:
            print("Merge Phase")
            phase = "merge"
        elif phase == 2:
            print("Systems Phase")
            phase = "systems"
        elif phase == 3:
            print("Online Phase")
            phase = "online"
    except:
        phase="offline"
        print("Phase default to offline")
    
# Create Mesh
    widths = [10., 80., 10.]
    nrefs = [10, 50, 10]
    Nmat = len(widths)
    nodes = [0.]
    for imat in range(Nmat):
        dx = widths[imat] / nrefs[imat]
        for i in range(nrefs[imat]):
            nodes.append(nodes[-1] + dx)
    meshgen = OrthogonalMeshGenerator(node_sets=[nodes])
    grid = meshgen.Execute()
    grid.SetUniformBlockID(0)

    # Set block IDs
    lv = RPPLogicalVolume(infx=True, infy=True, zmin=10.0, zmax=90.0)
    grid.SetBlockIDFromLogicalVolume(lv, 1, True)

    num_groups = 1
    scatt = MultiGroupXS()
    scatt.LoadFromOpenSn("data/scatterer.xs")

    fissile = MultiGroupXS()
    fissile.LoadFromOpenSn("data/fissile.xs")

    # Angle information
    n_angles = 32  # Number of discrete angles
    scat_order = 0  # Scattering order

    pquad = GLProductQuadrature1DSlab(n_polar=n_angles,
                    scattering_order=scat_order)

    # k-eigenvalue iteration parameters
    kes_max_iterations = 5000
    kes_tolerance = 1e-8

    # Source iteration parameters
    si_max_iterations = 500
    si_tolerance = 1e-8

    # Delayed neutrons
    use_precursors = True

    # Create and configure the discrete ordinates solver
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
            {"block_ids": [1], "xs": fissile}
        ],
        scattering_order=0,
        options={
            "use_precursors": False,
            "verbose_inner_iterations": False,
            "verbose_outer_iterations": True,
        }
    )

    if phase == "online":
        rom_options = {
                "param_id": 0,
                "phase": phase,
                "param_file": "data/params.txt",
                "new_point": [nu, sigma_f]
            }
    else:
        rom_options = {
                "param_id": p_id,
                "phase": phase
            }

    rom = ROMProblem(problem=phys,options=rom_options)

    # Initialize and execute solver
    k_solver = PowerIterationROMSolver(problem=phys, rom_problem=rom)
    k_solver.Initialize()
    k_solver.Execute()
    
    if phase == "online":
        phys.WriteFluxMoments("output/rom")
    if phase == "offline":
        phys.WriteFluxMoments("output/fom")