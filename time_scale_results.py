import json
import numpy as np
import glob
import subprocess
import sys
import os
import re
import matplotlib.pyplot as plt

env = os.environ.copy()
env["PYTHONPATH"] = os.pathsep.join(sys.path)

config_file_path = os.path.join('.', 'ensaios', 'ponto', 'ponto_sem_plots.json')
temp_json = os.path.join('.', 'ensaios', 'ponto', 'ponto_temp.json')

with open(config_file_path, 'r') as f:
    data = json.load(f)

courant = 0.5
base_grid = 500
vel = data["specimen_params"]["cp"]

data["simul_configs"]["n_iter"] = 25
data["simul_configs"]["save_results"] = 0
data["simul_configs"]["show_anim"] = 0

files = [f for f in glob.glob("sim_*.py") if not os.path.basename(f).startswith("sim_cpu_")]

num_files = len(files)
colors = plt.get_cmap('tab20', num_files)

plt.figure(figsize=(16, 9))
xrange = np.arange(1, 11)
x_ticks = xrange * base_grid

for idx, file in enumerate(files):

    match = re.search(r"sim_(.*?)\.py", file).group(1)

    all_mean = []

    for i in xrange:

        width = 1000.0 * i
        w_len = int(base_grid * i)

        dx = width/w_len

        one_dx = 1 / dx

        # courant_number = cp_max * self._dt * np.sqrt(self._one_dx ** 2 + self._one_dy ** 2)

        data["roi"]["h_len"] = w_len
        data["roi"]["w_len"] = w_len
        data["roi"]["height"] = width
        data["roi"]["width"] = width

        data["simul_params"]["dt"] = courant / (np.sqrt(one_dx ** 2 + one_dx ** 2) * vel)

        with open(temp_json, 'w') as f:
            json.dump(data, f)

        print("Running: ", file)
        print("Iter: ", i)
        print("ROI size: ", w_len)
        resultado = subprocess.run(
            [
                sys.executable,
                file,
                "-c",
                temp_json
            ],
            capture_output=True,
            text=True,
            env=env
        )

        print("STDOUT:\n", resultado.stdout)
        print("STDERR:\n", resultado.stderr)

        tempo_medio_match = re.search(r"Tempo medio total \(inclui transferencia de dados\):\s*([\d\.]+)s", resultado.stdout)

        mean_time = float(tempo_medio_match.group(1)) if tempo_medio_match else None

        print("MEAN: ", mean_time)

        all_mean.append(mean_time)

    plt.plot(x_ticks, all_mean, 'o-', label=match, color=colors(idx))


plt.xlabel("Grid Size (Number of Points)")
plt.ylabel("Mean time (s)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("execution_times_gridpoints.png")
plt.show()