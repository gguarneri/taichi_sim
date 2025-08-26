import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
import subprocess
import re

data = "nsys_data"
files = glob.glob(os.path.join(data, "perfil_sim_*.nsys-rep"))

all_data = []

for file in files:
    print(f"Analising {file}")
    profile_cmd = [
        "nsys",
        "stats",
        file,
        "--report",
        "cuda_gpu_mem_size_sum",
        "--force-export=true"
    ]

    resultado = subprocess.run(
        profile_cmd,
        capture_output=True,
        text=True,
        encoding="latin-1",
        errors="replace"
    )

    lines = resultado.stdout.splitlines()

    start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("Total (MB)"):
            start = i
            break
    if start is None:
        print("Tabela não encontrada no output!")
        continue

    header = [
        "Total (MB)",
        "Count",
        "Avg (MB)",
        "Med (MB)",
        "Min (MB)",
        "Max (MB)",
        "StdDev (MB)",
        "Operation"
    ]

    data_lines = []
    for line in lines[start+2:]:
        if not line.strip():
            break
        parts = re.split(r"\s{2,}", line.strip())
        if len(parts) == 8:
            parts.append(os.path.basename(file))
            data_lines.append(parts)

    df = pd.DataFrame(data_lines, columns=header + ["Profile"])

    for col in header[:-1]:
        df[col] = df[col].str.replace(",", ".").astype(float)

    all_data.append(df)

big_df = pd.concat(all_data, ignore_index=True)

for operation, df_op in big_df.groupby("Operation"):
    plt.figure(figsize=(8, 5))
    plt.bar(df_op["Profile"], df_op["Total (MB)"])
    plt.ylabel("Total (MB)")
    plt.title(f"GPU MemOps Summary - {operation}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()