import os
import subprocess


class JobManager:
    """
    System-aware launcher for OpenSn.
    Decides how to run OpenSn (srun / mpirun / direct) based on environment.
    """

    def __init__(self, system="auto", opensn_exe=None):
        self.system = system
        self.opensn_exe = opensn_exe

    # -------------------------
    # System detection
    # -------------------------
    def detect_system(self):
        if self.system != "auto":
            return self.system

        # Slurm environment detection
        if os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_CLUSTER_NAME"):
            return "slurm"

        return "local"

    # -------------------------
    # Command construction
    # -------------------------
    def build_command(self, input_file, nprocs=1, launcher_args=None, opensn_args=None):
        exe = self.opensn_exe

        tail = [exe, "-i", str(input_file)]
        if opensn_args:
            tail.extend(opensn_args)

        if nprocs > 1:
            launcher = "mpirun"
        else:
            launcher = "none"

        cmd = []

        if launcher == "none":
            cmd = tail

        elif launcher == "mpirun":
            cmd = ["mpirun", "-np", str(nprocs)]
            if launcher_args:
                cmd.extend(launcher_args)
            cmd.extend(tail)

        return cmd

    # -------------------------
    # Execution
    # -------------------------
    def run(
        self,
        input_file,
        nprocs=1,
        workdir=None,
        launcher_args=None,
        opensn_args=None,
        stream_output=True,
        check=False,
    ):
        cmd = self.build_command(
            input_file=input_file,
            nprocs=nprocs,
            launcher_args=launcher_args,
            opensn_args=opensn_args,
        )
        print("OPENSN CMD:", cmd, flush=True)
        if stream_output:
            p = subprocess.run(cmd,cwd=workdir)
        else:
            p = subprocess.run(cmd,cwd=workdir,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
		text=True)