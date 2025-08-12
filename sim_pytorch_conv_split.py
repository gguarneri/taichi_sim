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
class SimulatorPytorchConvSplit(Simulator):
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
        sens_vx = torch.zeros((self._n_steps, self._n_rec), dtype=torch.float32, device=device)
        sens_vy = torch.zeros((self._n_steps, self._n_rec), dtype=torch.float32, device=device)
        sens_pressure = torch.zeros((self._n_steps, self._n_rec), dtype=torch.float32, device=device)

        # Calculo dos indices para o staggered grid
        ord = self._coefs.shape[0]
        idx_fd = np.array([[c + 1, c, -c, -c - 1] for c in range(ord)], dtype=int32)
        last = ord - 1

        # Definicao dos limites para a plotagem dos campos
        v_max = 100.0
        v_min = - v_max
        ix_min = self._roi.get_ix_min()
        ix_max = self._roi.get_ix_max()
        iy_min = self._roi.get_iz_min()
        iy_max = self._roi.get_iz_max()

        # Inicializa os mapas dos parametros de Lame
        kappa = (self._rho_grid_vx * self._cp_grid_vx * self._cp_grid_vx * self._dt)
        kappa = torch.tensor(kappa, device=device, dtype=torch.float32)
        h = self._dt * self._one_dx * self._one_dy

        # Cria o kernel do filtro para o calculo das derivadas parciais
        kernel_base = np.concatenate((self._coefs[::-1], -self._coefs))[:, np.newaxis]
        x_kernel = kernel_base * self._one_dx
        y_kernel = kernel_base.T * self._one_dy
        
        x_kernel = torch.tensor(x_kernel, device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        y_kernel = torch.tensor(y_kernel, device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        def conv2d_valid(x, kernel):
            x4d = x.unsqueeze(0).unsqueeze(0)
            out = F.conv2d(x4d, kernel, padding='valid')
            return out.squeeze(0).squeeze(0)
        
        # Acrescenta eixo se source_term for array unidimensional
        if self._n_pto_src == 1:
            source_term = self._source_term[:, np.newaxis]
        
        source_term = torch.tensor(source_term, device=device)

        # Inicio do laco de tempo
        t_gpu = time()
        with torch.no_grad():
            for it in range(1, self._n_steps + 1):
                # Calculo da pressao
                # Primeiro "laco" i: 1,NX-1; j: 2,NY -> [1:-2, 2:-1]
                i_dix = idx_fd[last, 1]
                i_dfx = idx_fd[last, 3]
                i_diy = idx_fd[last, 0]
                i_dfy = idx_fd[last, 2]

                value_dvx_dx[i_dix:i_dfx, :] = conv2d_valid(vx, x_kernel)
                value_dvy_dy[:, i_diy:i_dfy] = conv2d_valid(vy, y_kernel)

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
                pressure.data.add_(kappa * (value_dvx_dx + value_dvy_dy))
                
                # Adicao das fontes no campo de pressao
                for _isrc in range(self._n_pto_src):
                    pressure[self._ix_src[_isrc], self._iy_src[_isrc]] += (source_term[it - 1, _isrc] * h)

                # Calculo da velocidade
                # Primeiro "laco" i: 2,NX; j: 2,NY -> [2:-1, 2:-1]
                i_dix = idx_fd[last, 0]
                i_dfx = idx_fd[last, 2]
                i_diy = idx_fd[last, 0]
                i_dfy = idx_fd[last, 2]

                value_dpressure_dx[i_dix:i_dfx, :] = conv2d_valid(pressure, x_kernel)

                memory_dpressure_dx[i_dix:i_dfx, i_diy:i_dfy] = (
                        b_x[1:, :] * memory_dpressure_dx[i_dix:i_dfx, i_diy:i_dfy] +
                        a_x[1:, :] * value_dpressure_dx[i_dix:i_dfx, i_diy:i_dfy])

                value_dpressure_dx[i_dix:i_dfx, i_diy:i_dfy] = (
                        value_dpressure_dx[i_dix:i_dfx, i_diy:i_dfy] / k_x[1:, :] +
                        memory_dpressure_dx[i_dix:i_dfx, i_diy:i_dfy])

                vx += self._dt * (value_dpressure_dx / rho_grid_vx)

                # segunda parte:  i: 1,NX-1; j: 1,NY-1 -> [1:-2, 1:-2]
                i_dix = idx_fd[last, 1]
                i_dfx = idx_fd[last, 3]
                i_diy = idx_fd[last, 1]
                i_dfy = idx_fd[last, 3]

                value_dpressure_dy[:, i_diy:i_dfy] = conv2d_valid(pressure, y_kernel)

                memory_dpressure_dy[i_dix:i_dfx, i_diy:i_dfy] = (
                        b_y_half[:, :-1] * memory_dpressure_dy[i_dix:i_dfx, i_diy:i_dfy] +
                        a_y_half[:, :-1] * value_dpressure_dy[i_dix:i_dfx, i_diy:i_dfy])

                value_dpressure_dy[i_dix:i_dfx, i_diy:i_dfy] = (
                        value_dpressure_dy[i_dix:i_dfx, i_diy:i_dfy] / k_y_half[:, :-1] +
                        memory_dpressure_dy[i_dix:i_dfx, i_diy:i_dfy])

                vy += self._dt * (value_dpressure_dy / rho_grid_vy)
                
                # Aplica as condicoes de Dirichlet
                # xmin
                vx[:(ord - 1), :] = 0.0
                vy[:(ord - 1), :] = 0.0

                # xmax
                vx[-(ord - 1):, :] = 0.0
                vy[-(ord - 1):, :] = 0.0

                # ymin
                vx[:, :(ord - 1)] = 0.0
                vy[:, :(ord - 1)] = 0.0

                # ymax
                vx[:, -(ord - 1):] = 0.0
                vy[:, -(ord - 1):] = 0.0

                # Armazena os sinais dos sensores
                for _i in range(self._idx_rec.shape[0]):
                    _irec = self._idx_rec[_i]
                    if it >= self._delay_recv[_irec]:
                        _x = self._ix_rec[_i]
                        _y = self._iy_rec[_i]
                        sens_vx[it - 1, _irec] += vx[_x, _y]
                        sens_vy[it - 1, _irec] += vy[_x, _y]
                        sens_pressure[it - 1, _irec] += pressure[_x, _y]

                psn2 = torch.max(torch.abs(pressure)).item()
                if (it % self._it_display) == 0 or it == 5:
                    if self._show_debug:
                        print(f"Time step {it} out of {self._n_steps}")
                        print(f"Max absolute value of pressure = {psn2}")

                    if self._show_anim:
                        self._windows_gpu[0].imv.setImage(pressure[ix_min:ix_max, iy_min:iy_max].cpu().numpy(), levels=[v_min, v_max])
                        self._app.processEvents()

                # Verifica a estabilidade da simulacao
                if psn2 > STABILITY_THRESHOLD:
                    raise StabilityError("Simulacao tornando-se instavel", psn2)
                
        sim_time = time() - t_gpu
        pressure = pressure.cpu().numpy()
        sens_vx = sens_vx.cpu().numpy()
        sens_vy = sens_vy.cpu().numpy()
        sens_pressure = sens_pressure.cpu().numpy()

        # --------------------------------------------
        # A funcao de implementacao do simulador deve retornar
        # um dicionario com as seguintes chaves:
        #   - "pressure": campo de pressao
        #   - "sens_pressure": sinais da pressao nos sensores
        #   - "gpu_str": string de identificacao da GPU utilizada na simulacao
        #   - "sim_time": tempo da simulacao, medido com a funcao time()
        #   - opcionalmente pode ter uma mensagem exclusiva da implementacao em "msg_impl"
        # --------------------------------------------
        return {"pressure": pressure, "sens_pressure": sens_pressure,
                "gpu_str": "Pytorch - conv split", "sim_time": sim_time}


# ----------------------------------------------------------
# Avaliacao dos parametros na linha de comando
# ----------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", help="Configuration file", default="config.json")
args = parser.parse_args()

# Cria a instancia do simulador
sim_instance = SimulatorPytorchConvSplit(args.config)

# Executa simulacao
try:
    sim_instance.run()

except KeyError as key:
    print(f"Chave {key} nao encontrada no arquivo de configuracao.")

except ValueError as value:
    print(value)