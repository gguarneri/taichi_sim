import glob
import subprocess
import os

output_dir = "nsys_data"
os.makedirs(output_dir, exist_ok=True)

python_exec = r"C:\Users\Victor\miniconda3\envs\python310\python.exe"

files = [f for f in glob.glob("sim_*.py") if not os.path.basename(f).startswith("sim_cpu_")]
print("Scripts encontrados (GPU):", files)

for file in files:
    base_name = os.path.splitext(os.path.basename(file))[0]
    rep_file = os.path.join(output_dir, f"perfil_{base_name}.nsys-rep")
    json_file = os.path.join(output_dir, f"perfil_{base_name}_json.json")

    print(f"\n### Profiling {file} com Nsight Systems ###")

    profile_cmd = [
        "nsys",
        "profile",
        "--trace=cuda,nvtx,vulkan",
        "--gpu-metrics-devices=all",
        "--cuda-memory-usage=true",
        "--sample=process-tree",
        "--force-overwrite=true",
        f"--output={os.path.splitext(rep_file)[0]}",
        python_exec,
        file,
        "-c",
        r".\ensaios\ponto\ponto_sem_plots_GPU.json"
    ]

    resultado = subprocess.run(profile_cmd, capture_output=True, text=True)
    print("STDOUT:\n", resultado.stdout)
    print("STDERR:\n", resultado.stderr)

    export_cmd = [
        "nsys",
        "export",
        "--type=json",
        f"--output={json_file}",
        rep_file,
        "--force-overwrite=true"
    ]

    resultado_export = subprocess.run(export_cmd, capture_output=True, text=True)
    print("Export JSON STDOUT:\n", resultado_export.stdout)
    print("Export JSON STDERR:\n", resultado_export.stderr)
