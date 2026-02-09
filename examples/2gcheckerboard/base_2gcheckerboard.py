#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 2D Transport test. Checkerboard https://doi.org/10.1016/j.jcp.2022.111525

import os
import sys
import math
import time
import numpy as np

if "opensn_console" not in globals():
    from mpi4py import MPI
    size = MPI.COMM_WORLD.size
    rank = MPI.COMM_WORLD.rank
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))
    from pyopensn.mesh import OrthogonalMeshGenerator
    from pyopensn.xs import MultiGroupXS
    from pyopensn.source import VolumetricSource
    from pyopensn.aquad import GLCProductQuadrature2DXY
    from pyopensn.solver import DiscreteOrdinatesProblem, SteadyStateSolver
    from pyopensn.fieldfunc import FieldFunctionInterpolationVolume
    from pyopensn.logvol import RPPLogicalVolume

if __name__ == "__main__":

    try:
        print("Parameter id = {}".format(p_id))
    except:
        p_id=0
        print("Parameter id = {}".format(p_id))

    print("{} phase".format(phase))

    # Check number of processors
    num_procs = 4
    if size != num_procs:
        sys.exit(f"Incorrect number of processors. Expected {num_procs} processors but got {size}.")

    # Setup mesh
    nodes = []
    N = 70
    L = 7
    xmin = 0
    dx = L / N
    for i in range(N + 1):
        nodes.append(xmin + i * dx)
    meshgen = OrthogonalMeshGenerator(node_sets=[nodes, nodes],
        partitioner=KBAGraphPartitioner(
            nx=2,
            ny=2,
            xcuts=[3.5],
            ycuts=[3.5],
        ))
    grid = meshgen.Execute()

    # Set background (Scatterer) block ID = 0
    grid.SetUniformBlockID(0)

    # Set Source (central red square from x=3 to x=4, y=3 to y=4) block ID = 1
    logvol_src = RPPLogicalVolume(xmin=3.0, xmax=4.0,
                                ymin=3.0, ymax=4.0,
                                infz=True)
    grid.SetBlockIDFromLogicalVolume(logvol_src, 1, True)

    # Set Absorbers (green 1x1 squares) block ID = 2
    absorber_centers = [
        (1,1), (3,1), (5,1),
        (2,2), (4,2),
        (1,3), (5,3),
        (2,4), (4,4),
        (1,5), (5,5)
    ]
    for xc, yc in absorber_centers:
        vol_abs = RPPLogicalVolume(
            xmin=xc+0.0, xmax=xc+1.0,
            ymin=yc+0.0, ymax=yc+1.0,
            infz=True
        )
        grid.SetBlockIDFromLogicalVolume(vol_abs, 2, True)

    num_groups = 2
    scatterer = MultiGroupXS()
    scatterer.LoadFromOpenSn("data/scatterer.xs")

    absorber = MultiGroupXS()
    absorber.LoadFromOpenSn("data/absorber.xs")

    strength = [1.0, 0.0]
    src1 = VolumetricSource(block_ids=[1], group_strength=strength)

    # Setup Physics
    pquad = GLCProductQuadrature2DXY(n_polar=4, n_azimuthal=32, scattering_order=0)

    if phase == "online":
        rom_options={
            "param_id":pid,
            "phase":phase,
            "param_file":"data/params.txt",
            "new_point":[p0, p1]
        }
    else:
        rom_options={
            "param_id":pid,
            "phase":phase
        }
    
    phys = DiscreteOrdinatesProblem(
        mesh=grid,
        num_groups=num_groups,
        groupsets=[
            {
                "groups_from_to": [0, 1],
                "angular_quadrature": pquad,
                "angle_aggregation_num_subsets": 1,
                "inner_linear_method": "petsc_gmres",
                "l_abs_tol": 1.0e-8,
                "l_max_its": 300,
                "gmres_restart_interval": 100,
            },
        ],
        xs_map=[
            {"block_ids": [0, 1], "xs": scatterer},
            {"block_ids": [2], "xs": absorber}
        ],
        volumetric_sources=[src1]
    )

    rom = ROMProblem(problem=phys,options=rom_options)

    ss_solver = SteadyStateROMSolver(problem=phys, rom_problem=rom)
    ss_solver.Initialize()
    ss_solver.Execute()

    try:
        if phase == "online" and saveh5:
            phys.WriteFluxMoments("output/rom_{}_".format(pid))
        if phase == "offline" and saveh5:
            phys.WriteFluxMoments("output/fom_{}_".format(pid))
    except:
        if phase == "online":
            phys.WriteFluxMoments("output/rom")
        if phase == "offline":
            phys.WriteFluxMoments("output/fom")