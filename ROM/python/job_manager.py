import subprocess


class JobManager:
    """
    Launcher for OpenSn using direct execution or mpirun.
    """

    def __init__(self, opensn_exe=None):
        self.opensn_exe = opensn_exe

    # -------------------------
    # Command construction
    # -------------------------
    def build_command(self, input_file, nprocs=1, opensn_args=None):
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
        opensn_args=None,
        stream_output=True,
        check=False,
    ):
        cmd = self.build_command(
            input_file=input_file,
            nprocs=nprocs,
            opensn_args=opensn_args,
        )
        print("OPENSN CMD:", cmd, flush=True)
        if stream_output:
            p = subprocess.run(cmd, cwd=workdir, check=check)
        else:
            p = subprocess.run(
                cmd,
                cwd=workdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=check,
            )
        return p
