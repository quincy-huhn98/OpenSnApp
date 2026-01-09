#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

if __name__ == "__main__":

    try:
        print("Parameter id = {}".format(pid))
    except:
        p_id=0
        print("Parameter id = {}".format(pid))

    print("{} phase".format(phase))

    widths = [4.6, 1.126152]
    nrefs = [50, 50]
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
    lv = RPPLogicalVolume(infx=True, infy=True, zmin=0.0, zmax=4.6)
    grid.SetBlockIDFromLogicalVolume(lv, 1, True)

    num_groups = 2
    scatt = MultiGroupXS()
    scatt.LoadFromOpenSn("data/H2O_mg.xs")

    fissile = MultiGroupXS()
    fissile.LoadFromOpenSn("data/URRb.xs")

    # Angle information
    n_angles = 32  # Number of discrete angles
    scat_order = 0  # Scattering order

    pquad = GLProductQuadrature1DSlab(n_polar=n_angles,
                    scattering_order=scat_order)

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
        boundary_conditions=[
            {"name": "zmin", "type": "reflecting"},
        ],
        options={
            "use_precursors": False,
            "verbose_inner_iterations": False,
            "verbose_outer_iterations": True,
        }
    )

    if phase == "online":
        new_point = [p0, p1, p2, p3, p4, p5, p6, p7]
        print(new_point)
        rom_options = {
                "param_id": pid,
                "phase": phase,
                "param_file": "data/params.txt",
                "new_point": new_point
            }
    else:
        rom_options = {
                "param_id": pid,
                "phase": phase
            }

    rom = ROMProblem(problem=phys,options=rom_options)

    # Initialize and execute solver
    k_solver = PowerIterationROMSolver(problem=phys, rom_problem=rom, k_tol=1.0e-7)
    k_solver.Initialize()
    k_solver.Execute()

    try:
        if phase == "online" and saveh5:
            phys.WriteFluxMoments("output/rom_{}_".format(pid))
            np.savetxt("output/rom_k_{}.txt".format(pid), [k_solver.GetEigenvalue()])
        if phase == "offline" and saveh5:
            phys.WriteFluxMoments("output/fom_{}_".format(pid))
            np.savetxt("output/fom_k_{}.txt".format(pid), [k_solver.GetEigenvalue()])
    except:
        if phase == "online":
            phys.WriteFluxMoments("output/rom")
        if phase == "offline":
            phys.WriteFluxMoments("output/fom")