"""
Junção dos scripts antigos em um só para fazer a captura de dados de execução.

Para cada ROI gera colunas mean/median/max tanto para memória (GB)
quanto para utilização de GPU (%). Além disso, para cada ROI gera uma coluna
'ROI {size} mean time' com o tempo médio extraído do stdout/result files.
"""
import os
import sys
import glob
import json
import re
import subprocess
import time
import pandas as pd

BASE_GRID = 1000                       # base_grid -> ROI sizes = BASE_GRID * xrange
XRANGE = range(1, 6)                   # 1..5 -> 1000,2000,3000,4000,5000
SAMPLE_INTERVAL_MS = 200               # nvidia-smi polling interval (ms)
CONFIG_FILE_TEMPLATE = os.path.join(".", "ensaios", "ponto", "ponto_sem_plots.json")
TEMP_CONFIG = os.path.join(".", "ensaios", "ponto", "ponto_temp_runtime.json")
RESULTS_DIR = "ensaios/ponto/results"
LOG_DIR = "logs"
OUTPUT_SUMMARY_TSV = "summary_gpu_memory_times.tsv"
OUTPUT_DETAILED_CSV = "detailed_results.csv"

os.makedirs(LOG_DIR, exist_ok=True)


def list_simulators(pattern="sim_*.py"):
    files = sorted([f for f in glob.glob(pattern) if "sim_cpu_" not in os.path.basename(f)])
    return files


def start_nvidia_smi_log(logfile, sample_interval_ms):
    cmd = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total",
        "--format=csv,noheader,nounits",
        "-lms",
        str(sample_interval_ms),
    ]
    logf = open(logfile, "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.DEVNULL)
    return proc


def stop_nvidia_smi(proc):
    if proc is None:
        return
    try:
        proc.terminate()
        try:
            proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            proc.kill()
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def parse_nvidia_log_csv(logfile, sample_interval_ms):

    if not os.path.exists(logfile):
        return pd.DataFrame(columns=["gpu_util", "mem_util", "mem_used_MB", "mem_total_MB", "mem_used_GB"])
    
    df = pd.read_csv(logfile, names=["gpu_util", "mem_util", "mem_used_MB", "mem_total_MB"], skip_blank_lines=True)

    for c in ["gpu_util", "mem_util", "mem_used_MB", "mem_total_MB"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["mem_used_GB"] = df["mem_used_MB"] / 1024.0
    df["time_s"] = (df.index * (sample_interval_ms / 1000.0)).astype(float)

    return df


def extract_from_stdout(stdout_text):

    tempo = None
    energia = None
    if not stdout_text:
        return tempo, energia

    tm = re.search(r"Tempo medio total \(inclui transferencia de dados\):\s*([\d\.eE\-\+]+)s", stdout_text)
    if tm:
        try:
            tempo = float(tm.group(1))
        except Exception:
            tempo = None

    en = re.search(r"Energia refletida \(sensor\):\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", stdout_text)
    if en:
        try:
            energia = float(en.group(1))
        except Exception:
            energia = None

    return tempo, energia


def extract_from_result_files(sim_name):
    pattern = os.path.join(RESULTS_DIR, f"result_*{sim_name}*__desc.txt")
    files = glob.glob(pattern)
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fh:
                txt = fh.read()
            tempo, energia = extract_from_stdout(txt)
            if tempo is not None or energia is not None:
                return tempo, energia
        except Exception:
            continue
    return None, None


def update_temp_config(base_config_path, out_path, roi_w_len, roi_h_len):
    with open(base_config_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    data["roi"]["w_len"] = int(roi_w_len)
    data["roi"]["h_len"] = int(roi_h_len)
    data["roi"]["width"] = int(roi_w_len)
    data["roi"]["height"] = int(roi_h_len)

    data["simul_configs"]["n_iter"] = 10
    data["simul_configs"]["save_results"] = 1
    data["simul_configs"]["show_anim"] = 0

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def run_simulator(script_path, config_path, env):
    cmd = [sys.executable, script_path, "-c", config_path]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)
        return proc.stdout, proc.stderr, proc.returncode
    except Exception as e:
        return "", f"Exception while running: {e}", 1


def main():
    sims = list_simulators()

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(sys.path)

    roi_sizes = [BASE_GRID * i for i in XRANGE]

    # montar colunas: para cada ROI teremos 6 colunas (3 memória, 3 gpu) + 1 coluna de tempo médio
    mem_cols = []
    gpu_cols = []
    time_cols = []
    for r in roi_sizes:
        mem_cols += [f"ROI {r} mean memory", f"ROI {r} median memory", f"ROI {r} max memory"]
        gpu_cols += [f"ROI {r} mean gpu", f"ROI {r} median gpu", f"ROI {r} max gpu"]
        time_cols += [f"ROI {r} mean time"]

    # ordem das colunas: simulador | mem_cols | gpu_cols | time_cols | Sensor energy reflected | Mean time (s)
    all_columns = ["simulador"] + mem_cols + gpu_cols + time_cols + ["Sensor energy reflected", "Mean time (s)"]

    results_rows = []
    detailed_rows = []

    for sim in sims:
        sim_name = os.path.splitext(os.path.basename(sim))[0]
        print(f"\n=== Simulador: {sim_name} ===")
        row = {"simulador": sim_name}
        mem_stats_list = []   # list of (mean, median, max)
        gpu_stats_list = []   # list of (mean, median, max)
        time_stats_list = []  # list of tempo (mean extracted)
        sensor_energy_any = None
        times_for_mean = []

        for i, roi_factor in enumerate(XRANGE, start=1):
            roi_size = BASE_GRID * roi_factor
            print(f"-> ROI {roi_size} (iter {i})")
            update_temp_config(CONFIG_FILE_TEMPLATE, TEMP_CONFIG, roi_w_len=roi_size, roi_h_len=roi_size)

            smi_log = os.path.join(LOG_DIR, f"smi_{sim_name}_roi{roi_size}.csv")
            smi_proc = start_nvidia_smi_log(smi_log, SAMPLE_INTERVAL_MS)

            start_ts = time.time()
            stdout, stderr, rc = run_simulator(sim, TEMP_CONFIG, env)
            end_ts = time.time()

            stop_nvidia_smi(smi_proc)
            time.sleep(0.1)

            df_gpu = parse_nvidia_log_csv(smi_log, SAMPLE_INTERVAL_MS)

            mem_used = df_gpu["mem_used_GB"].dropna()
            if not mem_used.empty:
                mem_mean_gb = float(mem_used.mean())
                mem_median_gb = float(mem_used.median())
                mem_max_gb = float(mem_used.max())
            else:
                mem_mean_gb = float("nan")
                mem_median_gb = float("nan")
                mem_max_gb = float("nan")

            gpu_util = df_gpu["gpu_util"].dropna()
            if not gpu_util.empty:
                gpu_mean = float(gpu_util.mean())
                gpu_median = float(gpu_util.median())
                gpu_max = float(gpu_util.max())
            else:
                gpu_mean = float("nan")
                gpu_median = float("nan")
                gpu_max = float("nan")

            tempo, energia = extract_from_stdout(stdout)
            if tempo is None or energia is None:
                t2, e2 = extract_from_result_files(sim_name)
                tempo = tempo if tempo is not None else t2
                energia = energia if energia is not None else e2

            if (tempo is None or energia is None) and stderr:
                t3, e3 = extract_from_stdout(stderr)
                tempo = tempo if tempo is not None else t3
                energia = energia if energia is not None else e3

            if tempo is None:
                tempo = float(end_ts - start_ts)

            if sensor_energy_any is None and energia is not None:
                sensor_energy_any = energia

            times_for_mean.append(tempo)
            time_stats_list.append(tempo)

            detailed_rows.append({
                "simulador": sim_name,
                "roi_size": roi_size,
                "mem_mean_GB": mem_mean_gb,
                "mem_median_GB": mem_median_gb,
                "mem_max_GB": mem_max_gb,
                "gpu_mean_pct": gpu_mean,
                "gpu_median_pct": gpu_median,
                "gpu_max_pct": gpu_max,
                "tempo_medio_s": tempo,
                "energia_sensor": energia,
                "smi_log": smi_log,
                "returncode": rc
            })

            mem_stats_list.append((mem_mean_gb, mem_median_gb, mem_max_gb))
            gpu_stats_list.append((gpu_mean, gpu_median, gpu_max))

            print(f"   mem mean/median/max (GB) = {mem_mean_gb:.6g}/{mem_median_gb:.6g}/{mem_max_gb:.6g}; "
                  f"gpu mean/median/max (%) = {gpu_mean:.6g}/{gpu_median:.6g}/{gpu_max:.6g}; tempo={tempo:.6g}")

        for idx, r in enumerate(roi_sizes):
            mean_v, median_v, max_v = mem_stats_list[idx] if idx < len(mem_stats_list) else (float("nan"), float("nan"), float("nan"))
            row[f"ROI {r} mean memory"] = mean_v
            row[f"ROI {r} median memory"] = median_v
            row[f"ROI {r} max memory"] = max_v

        for idx, r in enumerate(roi_sizes):
            mean_v, median_v, max_v = gpu_stats_list[idx] if idx < len(gpu_stats_list) else (float("nan"), float("nan"), float("nan"))
            row[f"ROI {r} mean gpu"] = mean_v
            row[f"ROI {r} median gpu"] = median_v
            row[f"ROI {r} max gpu"] = max_v

        for idx, r in enumerate(roi_sizes):
            tval = time_stats_list[idx] if idx < len(time_stats_list) else float("nan")
            row[f"ROI {r} mean time"] = tval

        row["Sensor energy reflected"] = sensor_energy_any if sensor_energy_any is not None else float("nan")
        row["Mean time (s)"] = float(pd.Series(times_for_mean).mean()) if times_for_mean else float("nan")

        results_rows.append(row)

    df_summary = pd.DataFrame(results_rows, columns=all_columns)
    df_summary.to_csv(OUTPUT_SUMMARY_TSV, sep="\t", index=False, float_format="%.9g")
    print(f"\nResumo salvo em: {OUTPUT_SUMMARY_TSV}")

    df_detailed = pd.DataFrame(detailed_rows)
    df_detailed.to_csv(OUTPUT_DETAILED_CSV, index=False, sep="\t", float_format="%.9g")
    print(f"Detalhes salvos em: {OUTPUT_DETAILED_CSV}")
    print("Logs de smi por execução em:", LOG_DIR)


if __name__ == "__main__":
    main()