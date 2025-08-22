import glob
import subprocess
import os
import signal
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re


def listar_scripts(pattern):
    return [
        f for f in glob.glob(pattern) if not os.path.basename(f).startswith("sim_cpu_")
    ]


def monitorar_gpu(log_file, sample_interval_ms):
    cmd = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total",
        "--format=csv,noheader,nounits",
        "-lms",
        str(sample_interval_ms),
    ]
    logf = open(log_file, "w")
    proc = subprocess.Popen(cmd, stdout=logf)
    return proc, logf


def executar_simulacao(script, config_file, python_exec):
    subprocess.run([python_exec, script, "-c", config_file], check=True)


def encerrar_monitoramento(proc, logf):
    os.kill(proc.pid, signal.SIGTERM)
    logf.close()


def processar_log_gpu(log_file, sample_interval_ms):
    df = pd.read_csv(
        log_file, names=["gpu_util", "mem_util", "mem_used_MB", "mem_total_MB"]
    )
    df["time_s"] = df.index * (sample_interval_ms / 1000)
    df["mem_used_GB"] = df["mem_used_MB"] / 1024
    return df


def plot_gpu_memoria(df, output_img, titulo):
    pico_mem = df["mem_used_GB"].max()
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.set_xlabel("Tempo (s)")
    ax1.set_ylabel("Memória usada (GB)", color="tab:blue")
    ax1.plot(
        df["time_s"],
        df["mem_used_GB"],
        color="tab:blue",
        label=f"Memória usada (pico={pico_mem:.2f} GB)",
    )
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_ylim(0, 1)
    ax2 = ax1.twinx()
    ax2.set_ylabel("Uso GPU (%)", color="tab:red")
    ax2.plot(
        df["time_s"], df["gpu_util"], color="tab:red", alpha=0.7, label="Uso GPU (%)"
    )
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2.set_ylim(0, 100)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    fig.suptitle(titulo)
    fig.tight_layout()
    plt.savefig(output_img)
    plt.close()
    return pico_mem


def plot_mean_std(data_list, labels, ylabel, output_file):
    fig, ax = plt.subplots(figsize=(12, 6))
    means = [np.mean(d) for d in data_list]
    mins = [np.min(d) for d in data_list]
    maxs = [np.max(d) for d in data_list]

    yerr = [np.array(means) - np.array(mins), np.array(maxs) - np.array(means)]

    x = np.arange(1, len(data_list) + 1)
    ax.plot(x, means, "o", color="red", label="Média")
    ax.errorbar(
        x,
        means,
        yerr=yerr,
        fmt="none",
        ecolor="green",
        elinewidth=2,
        capsize=5,
        label="Min-Máx",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Comparação de {ylabel} entre métodos")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def extrair_dados_resultados(arquivo):
    with open(arquivo, "r", encoding="utf-8") as f:
        texto = f.read()
    return {
        "simulador": re.search(r"Simulador:\s*(.+)", texto).group(1).strip(),
        "tempo_medio": float(
            re.search(r"Tempo medio de execucao:\s*([\d\.]+)s", texto).group(1)
        ),
        "tempo_std": float(
            re.search(
                r"Tempo medio de execucao:.*?Desvio padrao:\s*([\d\.eE\-]+)",
                texto,
                re.DOTALL,
            ).group(1)
        ),
        "mse_campo": float(
            re.search(r"MSE medio do campo de pressao:\s*([\d\.eE\-]+)", texto).group(1)
        ),
        "mse_campo_std": float(
            re.search(
                r"MSE medio do campo de pressao:.*?Desvio padrao:\s*([\d\.eE\-]+)",
                texto,
                re.DOTALL,
            ).group(1)
        ),
        "mse_sensores": float(
            re.search(
                r"MSE medio dos sensores de pressao:\s*([\d\.eE\-]+)", texto
            ).group(1)
        ),
        "mse_sensores_std": float(
            re.search(
                r"MSE medio dos sensores de pressao:.*?Desvio padrao:\s*([\d\.eE\-]+)",
                texto,
                re.DOTALL,
            ).group(1)
        ),
    }


def analisar_e_plotar_resultados(arquivos_txt, output_dir):
    df = pd.DataFrame([extrair_dados_resultados(arq) for arq in arquivos_txt])
    df = df.sort_values(by="simulador").reset_index(drop=True)

    fig, axs = plt.subplots(1, 3, figsize=(20, 6))

    axs[0].bar(
        df["simulador"],
        df["tempo_medio"],
        yerr=[df["tempo_medio"] - df["tempo_min"], df["tempo_max"] - df["tempo_medio"]],
        capsize=5,
        color="skyblue",
    )
    axs[0].set_title("Tempo médio de execução (s)")
    axs[0].set_ylabel("Tempo (segundos)")
    axs[0].tick_params(axis="x", rotation=45)

    # Campo
    axs[1].bar(
        df["simulador"],
        df["mse_campo"],
        yerr=[df["mse_campo"] - df["mse_campo_min"], df["mse_campo_max"] - df["mse_campo"]],
        capsize=5,
        color="lightgreen",
    )
    axs[1].set_title("MSE médio do campo de pressão")
    axs[1].set_ylabel("MSE (escala log)")
    axs[1].set_yscale("log")
    axs[1].tick_params(axis="x", rotation=45)

    axs[2].bar(
        df["simulador"],
        df["mse_sensores"],
        yerr=[df["mse_sensores"] - df["mse_sensores_min"], df["mse_sensores_max"] - df["mse_sensores"]],
        capsize=5,
        color="salmon",
    )
    axs[2].set_title("MSE médio dos sensores de pressão")
    axs[2].set_ylabel("MSE (escala log)")
    axs[2].set_yscale("log")
    axs[2].tick_params(axis="x", rotation=45)

    fig.suptitle("Comparação de Métricas de Simulação", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "comparison_plots.png"))
    plt.close()


if __name__ == "__main__":
    OUTPUT_DIR = "analise_completa"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    PYTHON_EXEC = r"C:\Users\Victor\miniconda3\envs\python310\python.exe"
    CONFIG_FILE = r".\ensaios\ponto\ponto_sem_plots.json"
    SCRIPTS_PATTERN = "sim_*.py"
    RESULT_FILES_PATTERN = r".\ensaios\ponto\results\result_*__desc.txt"
    SAMPLE_INTERVAL_MS = 200
    scripts = listar_scripts(SCRIPTS_PATTERN)
    box_data_mem, box_data_gpu, labels_list = [], [], []
    for script in scripts:
        base_name = os.path.splitext(os.path.basename(script))[0]
        log_file = os.path.join(OUTPUT_DIR, f"{base_name}_memoria.csv")
        output_img = os.path.join(OUTPUT_DIR, f"{base_name}_uso_memoria_gpu.png")
        nvidia_proc, logf = monitorar_gpu(log_file, SAMPLE_INTERVAL_MS)
        executar_simulacao(script, CONFIG_FILE, PYTHON_EXEC)
        encerrar_monitoramento(nvidia_proc, logf)
        df_gpu = processar_log_gpu(log_file, SAMPLE_INTERVAL_MS)
        plot_gpu_memoria(df_gpu, output_img, f"Uso de GPU - {base_name}")
        box_data_mem.append(df_gpu["mem_used_GB"])
        box_data_gpu.append(df_gpu["gpu_util"])
        labels_list.append(base_name)
    if box_data_mem and box_data_gpu:
        plot_mean_std(
            box_data_mem,
            labels_list,
            "Memória usada (GB)",
            os.path.join(OUTPUT_DIR, "mean_std_memoria.png"),
        )
        plot_mean_std(
            box_data_gpu,
            labels_list,
            "Uso GPU (%)",
            os.path.join(OUTPUT_DIR, "mean_std_gpu.png"),
        )
    arquivos_txt = glob.glob(RESULT_FILES_PATTERN)
    if arquivos_txt:
        analisar_e_plotar_resultados(arquivos_txt, OUTPUT_DIR)
