import subprocess
import os
from pathlib import Path
import tempfile


def run_remote(remote_host: str, remote_user: str, local_file: str):
    # Create a temporary directory on the remote machine
    remote_path = (
        subprocess.check_output(["ssh", f"{remote_user}@{remote_host}", "mktemp -d"])
        .decode()
        .strip()
    )

    try:
        # Copy the files to remote machine
        print(f"Copying local files to remote")
        subprocess.run(
            f"scp *.py *.csv nounlist.txt {remote_user}@{remote_host}:{remote_path}",
            shell=True,
            check=True,
        )

        # Run the script remotely
        print(f"\nRunning script...")
        subprocess.run(
            [
                "ssh",
                f"{remote_user}@{remote_host}",
                f"source ~/.zshrc && conda activate torch310 && cd {remote_path} && python3 {os.path.basename(local_file)}",
            ]
        )

        # Copy results back
        print(f"\nCopying results back")
        subprocess.run(["scp", f"{remote_user}@{remote_host}:{remote_path}/*.csv", "."])

    finally:
        # Clean up the temporary directory
        subprocess.run(["ssh", f"{remote_user}@{remote_host}", f"rm -rf {remote_path}"])


if __name__ == "__main__":
    # Configure these values
    REMOTE_HOST = "thursday"
    REMOTE_USER = "oli"

    # Run the script remotely
    run_remote(REMOTE_HOST, REMOTE_USER, "main.py")
