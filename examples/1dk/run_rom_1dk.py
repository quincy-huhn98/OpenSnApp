import numpy as np
import sys, os
sys.path.insert(0, os.path.realpath("../python"))

import plotting
import utils

# Sampling training points
bounds = [[1.9,2.1],[0.65,1.25]]
num_params = 200

params = utils.sample_parameter_space(bounds, num_params)

# OFFLINE PHASE
phase = 0

for i, param in enumerate(params):
    utils.update_fission_xs("fissile_base.txt", "data/fissile.xs", [params[i,0]], [params[i,1]])

    cmd = "mpiexec -n 2 ../../build/rom_app_exec -i base_1dk.py -p phase={} -p nu={} -p sigma_f={} -p p_id={}"\
                                                                    .format(phase,param[0], param[1],     i)
    utils.run_opensn(cmd)

# MERGE PHASE
phase = 1

cmd = "mpiexec -n 2 ../../build/rom_app_exec -i base_1dk.py -p phase={} -p p_id={}".format(phase, i)
utils.run_opensn(cmd)

plotting.plot_sv(num_groups=1)


# SYSTEMS PHASE
phase = 2

for i, param in enumerate(params):
    utils.update_fission_xs("fissile_base.txt", "data/fissile.xs", [params[i,0]], [params[i,1]])

    cmd = "mpiexec -n 2 ../../build/rom_app_exec -i base_1dk.py -p phase={} -p nu={} -p sigma_f={} -p p_id={}"\
                                                                    .format(phase,param[0], param[1],     i)
    utils.run_opensn(cmd)

np.savetxt("data/params.txt", params)

# Generate Test Data
test_1 = np.random.uniform(1.9,2.1,[10,1])
test_2 = np.random.uniform(0.65,1.25,[10,1])
test = np.append(test_1,test_2, axis=1)

errors = []
k_errors = []
speedups = []

for i, param in enumerate(test):
    # ONLINE PHASE
    phase = 3
    utils.update_fission_xs("fissile_base.txt", "data/fissile.xs", [param[0]], [param[1]])

    cmd = "mpiexec -n 2 ../../build/rom_app_exec -i base_1dk.py -p phase={} -p nu={} -p sigma_f={} -p p_id={}"\
                                                                    .format(phase,param[0],param[1],     i)
    utils.run_opensn(cmd)
    rom_time = np.loadtxt("results/online_time.txt")

    # Reference FOM solution
    phase = 0
    cmd = "mpiexec -n 2 ../../build/rom_app_exec -i base_1dk.py -p phase={} -p nu={} -p sigma_f={} -p p_id={}"\
                                                                    .format(phase,param[0],param[1],     i)
    utils.run_opensn(cmd)
    fom_time = np.loadtxt("results/offline_time.txt")

    error = plotting.plot_1d_eigenvector("output/fom{}.h5", "output/rom{}.h5", ranks=range(2), pid=i)
    k_error = np.abs(np.loadtxt("output/fom_k.txt") - np.loadtxt("output/rom_k.txt"))

    k_errors.append(k_error)
    errors.append(error)
    speedups.append(fom_time/rom_time)

print("Avg Eigenvector Error ", np.mean(errors))
np.savetxt("results/errors.txt", errors)
print("Avg k Error ", np.mean(k_errors)*1e5, "pcm")
np.savetxt("results/k_errors.txt", k_errors)
print("Avg Speedup ", np.mean(speedups))
np.savetxt("results/speedups.txt", speedups)