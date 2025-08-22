import json
import matplotlib.pyplot as plt
import glob
import os

json_dir = "nsys_data"
json_files = glob.glob(os.path.join(json_dir, "*_json.json"))

images_dir = "images"
os.makedirs(images_dir, exist_ok=True)

for json_file in json_files:
    times = []
    mem_transfer_MB = []

    with open(json_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ev = json.loads(line)
            if 'CudaEvent' in ev:
                ce = ev['CudaEvent']

                if 'memcpy' in ce:
                    size = int(ce['memcpy']['sizebytes'])
                    t_start = int(ce['startNs'])
                    t_end = int(ce['endNs'])

                    times.append(t_start / 1e6)
                    mem_transfer_MB.append(0)

                    times.append(t_start / 1e6)
                    mem_transfer_MB.append(size / 1e6)

                    times.append(t_end / 1e6)
                    mem_transfer_MB.append(size / 1e6)

                    times.append(t_end / 1e6)
                    mem_transfer_MB.append(0)

    plt.figure(figsize=(10,5))
    plt.plot(times, mem_transfer_MB, marker='o', linestyle='-')
    plt.xlabel("Tempo (ms)")
    plt.ylabel("Transferência de Memória (MB)")
    plt.title(f"memcpy - {os.path.basename(json_file)}")
    plt.grid(True)

plt.show()
