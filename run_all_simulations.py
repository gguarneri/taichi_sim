import glob
import subprocess
import sys
import os
import time

env = os.environ.copy()
env["PYTHONPATH"] = os.pathsep.join(sys.path)

files = glob.glob("sim_*.py")
print(files)

config_file_path = os.path.join('.', 'ensaios', 'ponto', 'ponto_sem_plots.json')

for file in files:
    if file.find("cpu") == -1:
        print("Running: ", file)
        resultado = subprocess.run(
            [
                sys.executable,
                file,
                "-c",
                config_file_path
            ],
            capture_output=True,
            text=True,
            env=env
        )

        print("STDOUT:\n", resultado.stdout)
        print("STDERR:\n", resultado.stderr)
        time.sleep(1)