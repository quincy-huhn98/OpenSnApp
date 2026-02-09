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
        print("Parameter id = {}".format(pid))
    except:
        p_id=0
        print("Parameter id = {}".format(pid))

    print("{} phase".format(phase))
    
    # Create Mesh
    widths = [2., 1., 2., 1., 2.]
    nrefs = [200, 200, 200, 200, 200]
    Nmat = len(widths)
    nodes = [0.]
    for imat in range(Nmat):
        dx = widths[imat] / nrefs[imat]
        for i in range(nrefs[imat]):
            nodes.append(nodes[-1] + dx)
    meshgen = OrthogonalMeshGenerator(node_sets=[nodes])
    grid = meshgen.Execute()

    # Set block IDs
    z_min = 0.0
    z_max = widths[1]
    for imat in range(Nmat):
        z_max = z_min + widths[imat]
        print("imat=", imat, ", zmin=", z_min, ", zmax=", z_max)
        lv = RPPLogicalVolume(infx=True, infy=True, zmin=z_min, zmax=z_max)
        grid.SetBlockIDFromLogicalVolume(lv, imat, True)
        z_min = z_max

    # Add cross sections to materials
    total = [50., 5., 0., 1., 1.]
    c = [0., 0., 0., p0, p0]
    xs_map = len(total) * [None]
    for imat in range(Nmat):
        xs_ = MultiGroupXS()
        xs_.CreateSimpleOneGroup(total[imat], c[imat])
        xs_map[imat] = {
            "block_ids": [imat], "xs": xs_,
        }

    # Create sources in 1st and 4th materials
    src0 = VolumetricSource(block_ids=[0], group_strength=[50.])
    src1 = VolumetricSource(block_ids=[3], group_strength=[p1])

    # Angular Quadrature
    gl_quad = GLProductQuadrature1DSlab(n_polar=128, scattering_order=0)

    # LBS block option
    num_groups = 1

    phys = DiscreteOrdinatesProblem(
        mesh=grid,
        num_groups=num_groups,
        groupsets=[
            {
                "groups_from_to": (0, num_groups - 1),
                "angular_quadrature": gl_quad,
                "inner_linear_method": "petsc_gmres",
                "l_abs_tol": 1.0e-9,
                "l_max_its": 300,
                "gmres_restart_interval": 30,
            },
        ],
        xs_map=xs_map,
        volumetric_sources= [src0, src1],
        boundary_conditions= [
                    {"name": "zmin", "type": "vacuum"},
                    {"name": "zmax", "type": "vacuum"}
        ]         
    )

    if phase == "online":
        rom_options = {
                "param_id": pid,
                "phase": phase,
                "param_file": "data/params.txt",
                "new_point": [p0, p1]
            }
    else:
        rom_options = {
                "param_id": pid,
                "phase": phase
            }

    rom = ROMProblem(problem=phys,options=rom_options)

    # Initialize and execute solver
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

    
