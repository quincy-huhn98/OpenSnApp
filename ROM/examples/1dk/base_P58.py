#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import os


def _collect_new_point():
    """Collect p0, p1, ... command-line parameters into a Python list.

    The online ROM may be interpolated either in the original parameter space
    or in a reduced active-coordinate space. Therefore the number of p-values
    is intentionally not hard-coded.
    """
    values = []
    i = 0
    while True:
        name = "p{}".format(i)
        if name not in globals():
            break
        values.append(float(globals()[name]))
        i += 1

    if not values:
        raise RuntimeError(
            "phase='online' requires at least one active/interpolation coordinate "
            "to be passed as p0, p1, ..."
        )

    return values


def _parameter_file(new_point):
    """Select physical or active training coordinates for interpolation."""
    active_file = "data/params_AS.txt"
    if os.path.exists(active_file):
        active_points = np.atleast_2d(np.loadtxt(active_file))
        if active_points.shape[1] == len(new_point):
            return active_file
    return "data/params.txt"


if __name__ == "__main__":

    print("Parameter id = {}".format(pid))

    print("{} phase".format(phase))

    widths = [4.6, 1.126152]
    nrefs = [8, 4] if globals().get("test_mode", False) else [500, 500]
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
    n_angles = 8 if globals().get("test_mode", False) else 128
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
        new_point = _collect_new_point()
        print(new_point)
        rom_options = {
            "param_id": pid,
            "phase": phase,
            "param_file": _parameter_file(new_point),
            "new_point": new_point
        }
    else:
        rom_options = {
            "param_id": pid,
            "phase": phase
        }

    rom = ROMProblem(problem=phys, options=rom_options)

    # Initialize and execute solver
    if globals().get("use_nlke", False):
        k_solver = NLKEigenROMSolver(
            problem=phys,
            rom_problem=rom,
            nl_max_its=20,
            l_max_its=50,
            l_abs_tol=1.0e-10,
            l_rel_tol=1.0e-10,
        )
    else:
        k_solver = PowerIterationROMSolver(problem=phys, rom_problem=rom, k_tol=1.0e-7)
    k_solver.Initialize()
    k_solver.Execute()

    if phase == "online" and saveh5:
        phys.WriteFluxMoments("output/rom_{}_".format(pid))
        np.savetxt("output/rom_k_{}.txt".format(pid), [k_solver.GetEigenvalue()])
    if phase == "mipod" and saveh5:
        phys.WriteFluxMoments("output/mipod_{}_".format(pid))
        np.savetxt("output/mipod_k_{}.txt".format(pid), [k_solver.GetEigenvalue()])
    if phase == "offline" and saveh5:
        phys.WriteFluxMoments("output/fom_{}_".format(pid))
        np.savetxt("output/test_fom_k_{}.txt".format(pid), [k_solver.GetEigenvalue()])
    elif phase == "offline":
        np.savetxt("output/fom_k_{}.txt".format(pid), [k_solver.GetEigenvalue()])
