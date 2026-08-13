#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 2D Transport test. Checkerboard https://doi.org/10.1016/j.jcp.2022.111525

import os
import sys

if "opensn_console" not in globals():
    from mpi4py import MPI
    size = MPI.COMM_WORLD.size
    rank = MPI.COMM_WORLD.rank
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))
    from pyopensn.mesh import OrthogonalMeshGenerator, PETScGraphPartitioner
    from pyopensn.xs import MultiGroupXS
    from pyopensn.source import VolumetricSource
    from pyopensn.aquad import GLCProductQuadrature2DXY
    from pyopensn.solver import DiscreteOrdinatesProblem
    from pyopensn.logvol import RPPLogicalVolume

if __name__ == "__main__":

    if "p4" in globals():
        print("5 Parameter Case q = {}".format(p4))
        int_point = [p0, p1, p2, p3, p4]
    else:
        p4 = 1.0
        print("4 Parameter Case q nominal = {}".format(p4))
        int_point = [p0, p1, p2, p3]

    print("Parameter id = {}".format(pid))

    print("{} phase".format(phase))

    # Setup mesh
    nodes = []
    N = 7 if globals().get("test_mode", False) else 70
    L = 7
    xmin = 0
    dx = L / N
    for i in range(N + 1):
        nodes.append(xmin + i * dx)
    meshgen = OrthogonalMeshGenerator(
        node_sets=[nodes, nodes],
        partitioner=PETScGraphPartitioner(type="parmetis"),
    )
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
        (1, 1), (3, 1), (5, 1),
        (2, 2), (4, 2),
        (1, 3), (5, 3),
        (2, 4), (4, 4),
        (1, 5), (5, 5)
    ]
    for xc, yc in absorber_centers:
        vol_abs = RPPLogicalVolume(
            xmin=xc + 0.0, xmax=xc + 1.0,
            ymin=yc + 0.0, ymax=yc + 1.0,
            infz=True
        )
        grid.SetBlockIDFromLogicalVolume(vol_abs, 2, True)

    scatt_t = p3 + p1
    num_groups = 1
    scatterer = MultiGroupXS()
    scatterer.CreateSimpleOneGroup(sigma_t=scatt_t, c=p1 / scatt_t)

    abs_t = p2 + p0

    absorber = MultiGroupXS()
    absorber.CreateSimpleOneGroup(sigma_t=abs_t, c=p0 / abs_t)

    strength = [0.0]
    src0 = VolumetricSource(block_ids=[0], group_strength=strength)
    strength = [p4]
    src1 = VolumetricSource(block_ids=[1], group_strength=strength)

    # Setup Physics
    n_polar = 2 if globals().get("test_mode", False) else 4
    n_azimuthal = 8 if globals().get("test_mode", False) else 32
    pquad = GLCProductQuadrature2DXY(
        n_polar=n_polar,
        n_azimuthal=n_azimuthal,
        scattering_order=0,
    )

    if phase == "online":
        rom_options = {
            "param_id": pid,
            "phase": phase,
            "param_file": "data/params.txt",
            "new_point": int_point
        }
    else:
        rom_options = {
            "param_id": pid,
            "phase": phase
        }

    phys = DiscreteOrdinatesProblem(
        mesh=grid,
        num_groups=num_groups,
        groupsets=[
            {
                "groups_from_to": [0, 0],
                "angular_quadrature": pquad,
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
        volumetric_sources=[src0, src1],
        sweep_type="CBC",
    )

    rom = ROMProblem(problem=phys, options=rom_options)

    ss_solver = SteadyStateROMSolver(problem=phys, rom_problem=rom)
    ss_solver.Initialize()
    ss_solver.Execute()

    if phase == "online" and saveh5:
        phys.WriteFluxMoments("output/rom_{}_".format(pid))
    if phase == "mipod" and saveh5:
        phys.WriteFluxMoments("output/mipod_{}_".format(pid))
    if phase == "offline" and saveh5:
        phys.WriteFluxMoments("output/fom_{}_".format(pid))
