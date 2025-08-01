# =======================
# Importacao de pacotes de uso geral
# =======================
import argparse
from time import time
import numpy as np
from sim_support import *
from sim_support.simulator import Simulator

# ======================
# Importacao de pacotes especificos para a implementacao do simulador
# ======================
import torch
import laplacian_cuda


# -----------------------------------------------------------------------------
# Aqui deve ser implementado o simulador como uma classe herdada de Simulator
# -----------------------------------------------------------------------------
class SimulatorCupyCuda(Simulator):
    def __init__(self, file_config):
        # Chama do construtor padrao, que le o arquivo de configuracao
        super().__init__(file_config)

        # Define o nome do simulador
        self._name = "Pytorch-CUDA"

    def implementation(self):
        super().implementation()

        # --------------------------------------------
        # Aqui comeca o codigo especifico do simulador
        # --------------------------------------------
        # Transfere arrays de parametros para a GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Usando dispositivo: {device}")

        dx = self._dx
        dz = self._dy
        dt = self._dt
        Nt = self._n_steps
        Nx = self._nx
        Nz = self._ny
        vp = torch.from_numpy(self._cp_grid_vx).to(device)

        # Campos
        u = torch.zeros(Nx, Nz, device=device)
        u_0 = torch.zeros_like(u)

        # Arrays para os sensores
        sens_pressure = np.zeros((self._n_steps, self._n_rec), dtype=flt32)

        # Definicao dos limites para a plotagem dos campos
        v_max = 100.0
        v_min = -v_max
        ix_min = self._roi.get_ix_min()
        ix_max = self._roi.get_ix_max()
        iy_min = self._roi.get_iz_min()
        iy_max = self._roi.get_iz_max()
        
        # Kernels
        def laplacian(u):
            return laplacian_cuda.laplacian(u.contiguous())

        # Acrescenta eixo se source_term for array unidimensional
        if self._n_pto_src == 1:
            self._source_term = self._source_term[:, np.newaxis]

        # Laco de tempo para execucao da simulacao
        t_sim_start = time()
        for n in range(Nt):
            lap_u = laplacian(u)

            u_1 = vp**2 * dt**2 / (dx*dz) * lap_u + 2 * u - u_0
            
            for _isrc in range(self._n_pto_src):
                u_1[self._ix_src[_isrc], self._iy_src[_isrc]] += self._source_term[n - 1, _isrc]

            u_0 = u
            u = u_1

            for _i in range(self._idx_rec.shape[0]):
                _irec = self._idx_rec[_i]
                if n >= self._delay_recv[_irec]:
                    _x = self._ix_rec[_i]
                    _y = self._iy_rec[_i]
                    sens_pressure[n - 1, _irec] += u[_x, _y]

            psn2 = torch.max(torch.abs(u))
            if (n % self._it_display) == 0 or n == 5:
                if self._show_debug:
                    print(f"Time step {n} out of {self._n_steps}")
                    print(f"Max pressure = {psn2}")

                if self._show_anim:
                    self._windows_gpu[-1].imv.setImage(
                        u[ix_min:ix_max, iy_min:iy_max].detach().cpu().numpy(),
                        levels=[v_min, v_max],
                    )
                    self._app.processEvents()

            # Verifica a estabilidade da simulacao
            if psn2 > STABILITY_THRESHOLD:
                raise StabilityError("Simulacao tornando-se instavel", psn2)

        t_sim_end = time()

        return {
            "vx": None,
            "vy": None,
            "pressure": u.detach().cpu().numpy(),
            "sens_vx": None,
            "sens_vy": None,
            "sens_pressure": sens_pressure,
            "gpu_str": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
            ),
            "sim_time": t_sim_end - t_sim_start,
            "msg_impl": "Torch-CPML",
        }


# ----------------------------------------------------------
# Avaliacao dos parametros na linha de comando
# ----------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", help="Configuration file", default="config.json")
args = parser.parse_args()

# Cria a instancia do simulador
sim_instance = SimulatorCupyCuda(args.config)

# Executa simulacao
try:
    sim_instance.run()

except KeyError as key:
    print(f"Chave {key} nao encontrada no arquivo de configuracao.")

except ValueError as value:
    print(value)
