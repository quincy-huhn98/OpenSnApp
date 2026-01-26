import numpy as np
import sys, os
sys.path.insert(0, os.path.realpath("../python"))
import matplotlib.pyplot as plt

import plotting
import utils
from active_subspace import *

x_nom = [0.35,0.35,0.3,0.5,0.5]

def evaluator(x):
    pid = np.loadtxt("data/pid.txt")
    utils.update_fission_xs("fissile_base.txt", "data/fissile.xs", [x[0]], [x[1]], [[x[2]]])
    utils.update_xs("scatterer_base.txt", "data/scatterer.xs", [x[3]], [[x[4]]])

    cmd = "mpiexec -n 2 ../../build/rom_app_exec -i base_1dk.py -p phase={} -p p_id={}"\
                                                                    .format(0, int(pid))
    utils.run_opensn(cmd)
    pid += 1
    np.savetxt("data/pid.txt", [pid])
    k_eff = np.loadtxt("output/fom_k.txt")
    # append to file
    with open("data/params.txt", "ab") as f:
        if x.ndim == 1:
            x = x.reshape(1, -1)
        np.savetxt(f, x)
    return k_eff

# Sampling training points
num_params = 50

np.savetxt("data/pid.txt", [0])
np.savetxt("data/params.txt",[])

normalizer, state, basis = create_subspace(x_nom, evaluator)

num_samples = int(num_params - np.loadtxt("data/pid.txt"))
batch = sample_subspace(num_samples, normalizer, basis.W1)

k_eff = []
for x in batch.x:
    k = evaluator(x)
    k_eff.append(k)

plt.plot(batch.z[:,0], k_eff, '.')
plt.savefig('output/AS.jpg')

i = int(num_params - 1)
# # OFFLINE PHASE
# phase = 0

# for i, param in enumerate(params):
#     utils.update_fission_xs("fissile_base.txt", "data/fissile.xs", [params[i,0]], [params[i,1]])

#     cmd = "mpiexec -n 2 ../../build/rom_app_exec -i base_1dk.py -p phase={} -p nu={} -p sigma_f={} -p p_id={}"\
#                                                                     .format(phase,param[0], param[1],     i)
#     utils.run_opensn(cmd)

params = np.loadtxt("data/params.txt")
# MERGE PHASE
phase = 1

cmd = "mpiexec -n 2 ../../build/rom_app_exec -i base_1dk.py -p phase={} -p p_id={}".format(phase, i)
utils.run_opensn(cmd)

plotting.plot_sv(num_groups=1)


# SYSTEMS PHASE
phase = 2

for i, param in enumerate(params):
    utils.update_fission_xs("fissile_base.txt", "data/fissile.xs", [params[i,0]], [params[i,1]], [[params[i,2]]])
    utils.update_xs("scatterer_base.txt", "data/scatterer.xs", [params[i,3]], [[params[i,4]]])

    cmd = "mpiexec -n 2 ../../build/rom_app_exec -i base_1dk.py -p phase={} -p p_id={}"\
                                                                    .format(phase,     i)
    utils.run_opensn(cmd)

np.savetxt("data/params.txt", params)

x_nom = [0.35,0.35,0.3,0.5,0.5]
# Generate Test Data
test_1 = np.random.uniform(x_nom[0]*0.8,x_nom[0]*1.2,[10,1])
test_2 = np.random.uniform(x_nom[1]*0.8,x_nom[1]*1.2,[10,1])
test_3 = np.random.uniform(x_nom[2]*0.8,x_nom[2]*1.2,[10,1])
test_4 = np.random.uniform(x_nom[3]*0.8,x_nom[3]*1.2,[10,1])
test_5 = np.random.uniform(x_nom[4]*0.8,x_nom[4]*1.2,[10,1])
test = np.append(test_1,test_2, axis=1)
test = np.append(test,test_3, axis=1)
test = np.append(test,test_4, axis=1)
test = np.append(test,test_5, axis=1)

test = [x_nom]
errors = []
k_errors = []
speedups = []

for i, param in enumerate(test):
    # ONLINE PHASE
    phase = 3
    utils.update_fission_xs("fissile_base.txt", "data/fissile.xs", [param[0]], [param[1]], [[param[2]]])
    utils.update_xs("scatterer_base.txt", "data/scatterer.xs", [param[3]], [[param[4]]])

    cmd = "mpiexec -n 2 ../../build/rom_app_exec -i base_1dk.py -p phase={} -p p1={} -p p2={} -p p3={} -p p4={} -p p5={} -p p_id={}"\
                                                                    .format(phase,param[0],param[1],param[2],param[3],param[4],     i)
    utils.run_opensn(cmd)
    rom_time = np.loadtxt("results/online_time.txt")

    # Reference FOM solution
    phase = 0
    cmd = "mpiexec -n 2 ../../build/rom_app_exec -i base_1dk.py -p phase={} -p p_id={}"\
                                                                    .format(phase,     i)
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