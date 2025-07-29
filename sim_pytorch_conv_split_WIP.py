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
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Aqui deve ser implementado o simulador como uma classe herdada de Simulator
# -----------------------------------------------------------------------------
class SimulatorCupyCuda(Simulator):
    def __init__(self, file_config):
        # Chama do construtor padrao, que le o arquivo de configuracao
        super().__init__(file_config)

        # Define o nome do simulador
        self._name = "Pytorch-conv"

    def implementation(self):
        super().implementation()

        # --------------------------------------------
        # Aqui comeca o codigo especifico do simulador
        # --------------------------------------------
        # Transfere arrays de parametros para a GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Usando dispositivo: {device}")

        # Arrays para as variaveis de memoria do calculo
        memory_dvx_dx = torch.zeros((self._nx, self._ny), dtype=torch.float32, device=device)
        memory_dvy_dy = torch.zeros((self._nx, self._ny), dtype=torch.float32, device=device)
        memory_dpressure_dx = torch.zeros((self._nx, self._ny), dtype=torch.float32, device=device)
        memory_dpressure_dy = torch.zeros((self._nx, self._ny), dtype=torch.float32, device=device)

        value_dvx_dx = torch.zeros((self._nx, self._ny), dtype=torch.float32, device=device)
        value_dvy_dy = torch.zeros((self._nx, self._ny), dtype=torch.float32, device=device)
        value_dpressure_dx = torch.zeros((self._nx, self._ny), dtype=torch.float32, device=device)
        value_dpressure_dy = torch.zeros((self._nx, self._ny), dtype=torch.float32, device=device)
        
        b_x_half = torch.tensor(self._b_x_half, device=device, dtype=torch.float32)
        a_x_half = torch.tensor(self._a_x_half, device=device, dtype=torch.float32)
        b_y = torch.tensor(self._b_y, device=device, dtype=torch.float32)
        a_y = torch.tensor(self._a_y, device=device, dtype=torch.float32)
        k_x_half = torch.tensor(self._k_x_half, device=device, dtype=torch.float32)
        k_y = torch.tensor(self._k_y, device=device, dtype=torch.float32)
        b_x = torch.tensor(self._b_x, device=device, dtype=torch.float32)
        a_x = torch.tensor(self._a_x, device=device, dtype=torch.float32)
        k_x = torch.tensor(self._k_x, device=device, dtype=torch.float32)
        rho_grid_vx = torch.tensor(self._rho_grid_vx, device=device, dtype=torch.float32)
        b_y_half = torch.tensor(self._b_y_half, device=device, dtype=torch.float32)
        a_y_half = torch.tensor(self._a_y_half, device=device, dtype=torch.float32)
        k_y_half = torch.tensor(self._k_y_half, device=device, dtype=torch.float32)
        rho_grid_vy = torch.tensor(self._rho_grid_vy, device=device, dtype=torch.float32)

        # Arrays dos campos de velocidade e pressoes
        vx = torch.zeros((self._nx, self._ny), dtype=torch.float32, device=device)
        vy = torch.zeros((self._nx, self._ny), dtype=torch.float32, device=device)
        pressure = torch.zeros((self._nx, self._ny), dtype=torch.float32, device=device)
        
        # Arrays para os sensores
        sens_vx = np.zeros((self._n_steps, self._n_rec), dtype=flt32)
        sens_vy = np.zeros((self._n_steps, self._n_rec), dtype=flt32)
        sens_pressure = np.zeros((self._n_steps, self._n_rec), dtype=flt32)

        # Calculo dos indices para o staggered grid
        ord = self._coefs.shape[0]
        idx_fd = torch.tensor([[c + ord,  # ini half grid
                            -c + ord - 1,  # ini full grid
                            c - ord + 1,  # fin half grid
                            -c - ord]  # fin full grid
                        for c in range(ord)], dtype=torch.int32)

        # Definicao dos limites para a plotagem dos campos
        v_max = 100.0
        v_min = - v_max
        ix_min = self._roi.get_ix_min()
        ix_max = self._roi.get_ix_max()
        iy_min = self._roi.get_iz_min()
        iy_max = self._roi.get_iz_max()

        # Inicializa os mapas dos parametros de Lame
        kappa = (self._rho_grid_vx * self._cp_grid_vx * self._cp_grid_vx *
                 self._dt * self._one_dx * self._one_dy)
        kappa = torch.tensor(kappa, device=device, dtype=torch.float32)

        # Cria o kernel do filtro para o calculo das derivadas parciais
        kernel_base = np.concatenate((self._coefs[::-1], -self._coefs))[:, np.newaxis]
        x_kernel = kernel_base * self._one_dx
        y_kernel = kernel_base.T * self._one_dy
        
        x_kernel = torch.tensor(x_kernel, device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        y_kernel = torch.tensor(y_kernel, device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        x_kernel = torch.flip(x_kernel, [2,3])
        y_kernel = torch.flip(y_kernel, [2,3])
        
        def conv2d_same(x, kernel):
            H, W = x.shape[-2:]

            x4d = x.unsqueeze(0).unsqueeze(0) if x.dim() == 2 else x
            k4d = kernel.unsqueeze(0).unsqueeze(0) if kernel.dim() == 2 else kernel

            kH, kW = k4d.shape[-2:]
            pad_y = kH // 2
            pad_x = kW // 2

            x_padded = F.pad(x4d, (pad_x, pad_x, pad_y, pad_y))

            k_flipped = torch.flip(k4d, [2, 3])

            out = F.conv2d(x_padded, k_flipped)
            out = out[..., :H, :W]
            return out.squeeze(0).squeeze(0)
        
        # Acrescenta eixo se source_term for array unidimensional
        if self._n_pto_src == 1:
            source_term = self._source_term[:, np.newaxis]
        
        source_term = torch.tensor(source_term, device=device)

        # Inicio do laco de tempo
        t_gpu = time()
        for it in range(1, self._n_steps + 1):
            # Calculo da pressao
            # Primeiro "laco" i: 1,NX-1; j: 2,NY -> [1:-2, 2:-1]
            i_dix = idx_fd[0, 1]
            i_dfx = idx_fd[0, 3]
            i_diy = idx_fd[0, 0]
            i_dfy = idx_fd[0, 2]

            value_dvx_dx[i_dix:i_dfx, i_diy:i_dfy] = conv2d_same(
                vx[i_dix + 1:i_dfx + 1, i_diy:i_dfy], x_kernel)
            
            value_dvy_dy[i_dix:i_dfx, i_diy:i_dfy] = conv2d_same(
                vy[i_dix:i_dfx, i_diy:i_dfy], y_kernel)

            memory_dvx_dx[i_dix:i_dfx, i_diy:i_dfy] = (
                    b_x_half[:-1, :] * memory_dvx_dx[i_dix:i_dfx, i_diy:i_dfy] +
                    a_x_half[:-1, :] * value_dvx_dx[i_dix:i_dfx, i_diy:i_dfy])
            memory_dvy_dy[i_dix:i_dfx, i_diy:i_dfy] = (
                    b_y[:, 1:] * memory_dvy_dy[i_dix:i_dfx, i_diy:i_dfy] +
                    a_y[:, 1:] * value_dvy_dy[i_dix:i_dfx, i_diy:i_dfy])

            value_dvx_dx[i_dix:i_dfx, i_diy:i_dfy] = (
                    value_dvx_dx[i_dix:i_dfx, i_diy:i_dfy] / k_x_half[:-1, :] +
                    memory_dvx_dx[i_dix:i_dfx, i_diy:i_dfy])
            value_dvy_dy[i_dix:i_dfx, i_diy:i_dfy] = (
                    value_dvy_dy[i_dix:i_dfx, i_diy:i_dfy] / k_y[:, 1:] +
                    memory_dvy_dy[i_dix:i_dfx, i_diy:i_dfy])

            # compute the pressure using the Lame parameters
            pressure += kappa * (value_dvx_dx + value_dvy_dy)
            
            # Adicao das fontes no campo de pressao
            for _isrc in range(self._n_pto_src):
                pressure[self._ix_src[_isrc], self._iy_src[_isrc]] += (source_term[it - 1, _isrc] * 
                                                                       self._dt * self._one_dx * self._one_dy)

            # Calculo da velocidade
            # Primeiro "laco" i: 2,NX; j: 2,NY -> [2:-1, 2:-1]
            i_dix = idx_fd[0, 0]
            i_dfx = idx_fd[0, 2]
            i_diy = idx_fd[0, 0]
            i_dfy = idx_fd[0, 2]

            value_dpressure_dx[i_dix:i_dfx, i_diy:i_dfy] = conv2d_same(
                pressure[i_dix:i_dfx, i_diy:i_dfy], x_kernel)

            memory_dpressure_dx[i_dix:i_dfx, i_diy:i_dfy] = (
                    b_x[1:, :] * memory_dpressure_dx[i_dix:i_dfx, i_diy:i_dfy] +
                    a_x[1:, :] * value_dpressure_dx[i_dix:i_dfx, i_diy:i_dfy])

            value_dpressure_dx[i_dix:i_dfx, i_diy:i_dfy] = (
                    value_dpressure_dx[i_dix:i_dfx, i_diy:i_dfy] / k_x[1:, :] +
                    memory_dpressure_dx[i_dix:i_dfx, i_diy:i_dfy])

            vx += self._dt * (value_dpressure_dx / rho_grid_vx)

            # segunda parte:  i: 1,NX-1; j: 1,NY-1 -> [1:-2, 1:-2]
            i_dix = idx_fd[0, 1]
            i_dfx = idx_fd[0, 3]
            i_diy = idx_fd[0, 1]
            i_dfy = idx_fd[0, 3]

            value_dpressure_dy[i_dix:i_dfx, i_diy:i_dfy] = conv2d_same(
                pressure[i_dix:i_dfx, i_diy + 1:i_dfy + 1], y_kernel)

            memory_dpressure_dy[i_dix:i_dfx, i_diy:i_dfy] = (
                    b_y_half[:, :-1] * memory_dpressure_dy[i_dix:i_dfx, i_diy:i_dfy] +
                    a_y_half[:, :-1] * value_dpressure_dy[i_dix:i_dfx, i_diy:i_dfy])

            value_dpressure_dy[i_dix:i_dfx, i_diy:i_dfy] = (
                    value_dpressure_dy[i_dix:i_dfx, i_diy:i_dfy] / k_y_half[:, :-1] +
                    memory_dpressure_dy[i_dix:i_dfx, i_diy:i_dfy])

            vy += self._dt * (value_dpressure_dy / rho_grid_vy)
            
            # Aplica as condicoes de Dirichlet
            # xmin
            vx[:ord - 1, :] = 0.0
            vy[:ord - 1, :] = 0.0

            # xmax
            vx[-ord - 1:, :] = 0.0
            vy[-ord - 1:, :] = 0.0

            # ymin
            vx[:, :ord - 1] = 0.0
            vy[:, :ord - 1] = 0.0

            # ymax
            vx[:, -ord - 1:] = 0.0
            vy[:, -ord - 1:] = 0.0

            # Armazena os sinais dos sensores
            for _i in range(self._idx_rec.shape[0]):
                _irec = self._idx_rec[_i]
                if it >= self._delay_recv[_irec]:
                    _x = self._ix_rec[_i]
                    _y = self._iy_rec[_i]
                    sens_vx[it - 1, _irec] += vx[_x, _y].cpu().numpy()
                    sens_vy[it - 1, _irec] += vy[_x, _y].cpu().numpy()
                    sens_pressure[it - 1, _irec] += pressure[_x, _y].cpu().numpy()

            psn2 = torch.max(torch.abs(pressure)).cpu().numpy().astype(np.float32)
            if (it % self._it_display) == 0 or it == 5:
                if self._show_debug:
                    print(f'Time step # {it} out of {self._n_steps}')
                    print(f'Max pressure = {psn2}')

                if self._show_anim:
                    self._windows_gpu[0].imv.setImage(pressure[ix_min:ix_max, iy_min:iy_max].cpu().numpy(), levels=[v_min, v_max])
                    self._app.processEvents()

            # Verifica a estabilidade da simulacao
            if psn2 > STABILITY_THRESHOLD:
                raise StabilityError("Simulacao tornando-se instavel", psn2)
                
        sim_time = time() - t_gpu

        return {
            "vx": None,
            "vy": None,
            "pressure": pressure.cpu().numpy(),
            "sens_vx": None,
            "sens_vy": None,
            "sens_pressure": sens_pressure,
            "gpu_str": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
            ),
            "sim_time": sim_time,
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