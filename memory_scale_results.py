import json
import numpy as np
import glob
import subprocess
import sys
import os
import re
import signal
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Configura ambiente
env = os.environ.copy()
env["PYTHONPATH"] = os.pathsep.join(sys.path)

config_file_path = os.path.join('.', 'ensaios', 'ponto', 'ponto_sem_plots.json')

with open(config_file_path, 'r') as f:
    data = json.load(f)

courant = 0.5
base_grid = 500
vel = data["specimen_params"]["cp"]

data["simul_configs"]["n_iter"] = 1
data["simul_configs"]["save_results"] = 0
data["simul_configs"]["show_anim"] = 0

files = [f for f in glob.glob("sim_*.py") if not os.path.basename(f).startswith("sim_cpu_")]

num_files = len(files)
colors = cm.get_cmap('tab20', num_files)

# Funções auxiliares
def monitorar_gpu(log_file, sample_interval_ms=200):
    cmd = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total",
        "--format=csv,noheader,nounits",
        "-lms", str(sample_interval_ms),
    ]
    logf = open(log_file, "w")
    proc = subprocess.Popen(cmd, stdout=logf)
    return proc, logf

def encerrar_monitoramento(proc, logf):
    os.kill(proc.pid, signal.SIGTERM)
    logf.close()

def processar_log_gpu(log_file, sample_interval_ms):
    df = pd.read_csv(
        log_file,
        names=["gpu_util", "mem_util", "mem_used_MB", "mem_total_MB"]
    )
    df["mem_used_GB"] = df["mem_used_MB"] / 1024
    return df

# Pastas
os.makedirs("memory_scale_results", exist_ok=True)

# Definição de grid
xrange = np.arange(1, 11)
x_ticks = xrange * base_grid

# Figuras separadas
plt_mem = plt.figure(figsize=(16, 9))
ax_mem = plt_mem.add_subplot(111)

plt_gpu = plt.figure(figsize=(16, 9))
ax_gpu = plt_gpu.add_subplot(111)

for idx, file in enumerate(files):
    match = re.search(r"sim_(.*?)\.py", file).group(1)

    all_mean_mem = []
    all_mean_gpu = []

    for i in xrange:
        width = 1000.0 * i
        w_len = int(base_grid * i)
        dx = width / w_len
        one_dx = 1 / dx

        data["roi"]["h_len"] = w_len
        data["roi"]["w_len"] = w_len
        data["roi"]["height"] = width
        data["roi"]["width"] = width
        data["simul_params"]["dt"] = courant / (np.sqrt(one_dx ** 2 + one_dx ** 2) * vel)

        config_temp = r".\ensaios\ponto\ponto_temp.json"
        with open(config_temp, 'w') as f:
            json.dump(data, f)

        log_file = os.path.join("memory_scale_results", f"gpu_log_{match}_{i}.csv")
        proc, logf = monitorar_gpu(log_file)

        print(f"Rodando {file} - Iter {i} (w_len={w_len})")
        subprocess.run([sys.executable, file, "-c", config_temp], env=env)

        encerrar_monitoramento(proc, logf)

        df_gpu = processar_log_gpu(log_file, 200)

        all_mean_mem.append(df_gpu["mem_used_GB"].mean())
        all_mean_gpu.append(df_gpu["gpu_util"].mean())

    # Adiciona às curvas
    ax_mem.plot(x_ticks, all_mean_mem, 'o-', label=f"{match}", color=colors(idx))
    ax_gpu.plot(x_ticks, all_mean_gpu, 's--', label=f"{match}", color=colors(idx))

# Ajustes VRAM
ax_mem.set_xlabel("Grid Size (Number of Points)")
ax_mem.set_ylabel("Consumo médio de VRAM (GB)")
ax_mem.legend()
ax_mem.grid()
plt_mem.tight_layout()
plt_mem.savefig(os.path.join("memory_scale_results", "vram_vs_gridpoints.png"))

# Ajustes GPU
ax_gpu.set_xlabel("Grid Size (Number of Points)")
ax_gpu.set_ylabel("Uso médio da GPU (%)")
ax_gpu.legend()
ax_gpu.grid()
plt_gpu.tight_layout()
plt_gpu.savefig(os.path.join("memory_scale_results", "gpu_usage_vs_gridpoints.png"))

plt.show()
