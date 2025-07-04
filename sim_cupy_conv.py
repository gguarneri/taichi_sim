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
import cupy
from cupyx.scipy import signal


# -----------------------------------------------------------------------------
# Aqui deve ser implementado o simulador como uma classe herdada de Simulator
# -----------------------------------------------------------------------------
class SimulatorCupyConv(Simulator):
    def __init__(self, file_config):
        # Chama do construtor padrao, que le o arquivo de configuracao
        super().__init__(file_config)
        
        # Define o nome do simulador
        self._name = "CuPy-conv"
        
        
    def implementation(self):
        super().implementation()
        
        # --------------------------------------------
        # Aqui comeca o codigo especifico do simulador
        # --------------------------------------------
        # Transfere arrays de parametros para a GPU
        a_x_gpu = cupy.asarray(self._a_x)
        b_x_gpu = cupy.asarray(self._b_x)
        k_x_gpu = cupy.asarray(self._k_x)
        a_x_half_gpu = cupy.asarray(self._a_x_half)
        b_x_half_gpu = cupy.asarray(self._b_x_half)
        k_x_half_gpu = cupy.asarray(self._k_x_half)
        a_y_gpu = cupy.asarray(self._a_y)
        b_y_gpu = cupy.asarray(self._b_y)
        k_y_gpu = cupy.asarray(self._k_y)
        a_y_half_gpu = cupy.asarray(self._a_y_half)
        b_y_half_gpu = cupy.asarray(self._b_y_half)
        k_y_half_gpu = cupy.asarray(self._k_y_half)
        
        rho_grid_vx_gpu = cupy.asarray(self._rho_grid_vx)
        rho_grid_vy_gpu = cupy.asarray(self._rho_grid_vy)

        # Arrays para as variaveis de memoria do calculo
        memory_dvx_dx_gpu = cupy.zeros((self._nx, self._ny), dtype=cupy.float32)
        memory_dvy_dy_gpu = cupy.zeros((self._nx, self._ny), dtype=cupy.float32)
        memory_dpressure_dx_gpu = cupy.zeros((self._nx, self._ny), dtype=cupy.float32)
        memory_dpressure_dy_gpu = cupy.zeros((self._nx, self._ny), dtype=cupy.float32)

        value_dvx_dx_gpu = cupy.zeros((self._nx, self._ny), dtype=cupy.float32)
        value_dvy_dy_gpu = cupy.zeros((self._nx, self._ny), dtype=cupy.float32)
        value_dpressure_dx_gpu = cupy.zeros((self._nx, self._ny), dtype=cupy.float32)
        value_dpressure_dy_gpu = cupy.zeros((self._nx, self._ny), dtype=cupy.float32)

        # Arrays dos campos de velocidade e pressoes
        vx_gpu = cupy.zeros((self._nx, self._ny), dtype=cupy.float32)
        vy_gpu = cupy.zeros((self._nx, self._ny), dtype=cupy.float32)
        pressure_gpu = cupy.zeros((self._nx, self._ny), dtype=cupy.float32)
        
        # Arrays para os sensores
        sens_vx = np.zeros((self._n_steps, self._n_rec), dtype=flt32)
        sens_vy = np.zeros((self._n_steps, self._n_rec), dtype=flt32)
        sens_pressure = np.zeros((self._n_steps, self._n_rec), dtype=flt32)

        # Calculo dos indices para o staggered grid
        ord = self._coefs.shape[0]
        idx_fd = np.array([[c + ord,  # ini half grid
                            -c + ord - 1,  # ini full grid
                            c - ord + 1,  # fin half grid
                            -c - ord]  # fin full grid
                        for c in range(ord)], dtype=np.int32)

        # Definicao dos limites para a plotagem dos campos
        v_max = 100.0
        v_min = - v_max
        ix_min = self._roi.get_ix_min()
        ix_max = self._roi.get_ix_max()
        iy_min = self._roi.get_iz_min()
        iy_max = self._roi.get_iz_max()

        # Inicializa os mapas dos parametros de Lame
        kappa_gpu = cupy.asarray(self._rho_grid_vx * self._cp_grid_vx * self._cp_grid_vx *
                                 self._dt * self._one_dx * self._one_dy)

        # Cria o kernel do filtro para o calculo das derivadas parciais
        x_kernel = np.concatenate((self._coefs[::-1], -self._coefs))[:, np.newaxis]
        x_kernel_gpu = cupy.asarray(x_kernel * self._one_dx)
        y_kernel_gpu = cupy.asarray(x_kernel.T * self._one_dy)

        # Acrescenta eixo se source_term for array unidimensional
        if self._n_pto_src == 1:
            self._source_term = self._source_term[:, np.newaxis]
            
        # Laco de tempo para execucao da simulacao
        t_gpu = time()
        for it in range(1, self._n_steps + 1):
            # Calculo da pressao
            # Primeiro "laco" i: 1,NX-1; j: 2,NY -> [1:-2, 2:-1]
            i_dix = idx_fd[0, 1]
            i_dfx = idx_fd[0, 3]
            i_diy = idx_fd[0, 0]
            i_dfy = idx_fd[0, 2]

            value_dvx_dx_gpu[i_dix:i_dfx, i_diy:i_dfy] = signal.convolve2d(
                vx_gpu[i_dix + 1:i_dfx + 1, i_diy:i_dfy], x_kernel_gpu, mode='same')
            value_dvy_dy_gpu[i_dix:i_dfx, i_diy:i_dfy] = signal.convolve2d(
                vy_gpu[i_dix:i_dfx, i_diy:i_dfy], y_kernel_gpu, mode='same')

            memory_dvx_dx_gpu[i_dix:i_dfx, i_diy:i_dfy] = (
                    b_x_half_gpu[:-1, :] * memory_dvx_dx_gpu[i_dix:i_dfx, i_diy:i_dfy] +
                    a_x_half_gpu[:-1, :] * value_dvx_dx_gpu[i_dix:i_dfx, i_diy:i_dfy])
            memory_dvy_dy_gpu[i_dix:i_dfx, i_diy:i_dfy] = (
                    b_y_gpu[:, 1:] * memory_dvy_dy_gpu[i_dix:i_dfx, i_diy:i_dfy] +
                    a_y_gpu[:, 1:] * value_dvy_dy_gpu[i_dix:i_dfx, i_diy:i_dfy])

            value_dvx_dx_gpu[i_dix:i_dfx, i_diy:i_dfy] = (
                    value_dvx_dx_gpu[i_dix:i_dfx, i_diy:i_dfy] / k_x_half_gpu[:-1, :] +
                    memory_dvx_dx_gpu[i_dix:i_dfx, i_diy:i_dfy])
            value_dvy_dy_gpu[i_dix:i_dfx, i_diy:i_dfy] = (
                    value_dvy_dy_gpu[i_dix:i_dfx, i_diy:i_dfy] / k_y_gpu[:, 1:] +
                    memory_dvy_dy_gpu[i_dix:i_dfx, i_diy:i_dfy])

            # compute the pressure using the Lame parameters
            pressure_gpu = pressure_gpu + kappa_gpu * (value_dvx_dx_gpu + value_dvy_dy_gpu)

            # Calculo da velocidade
            # Primeiro "laco" i: 2,NX; j: 2,NY -> [2:-1, 2:-1]
            i_dix = idx_fd[0, 0]
            i_dfx = idx_fd[0, 2]
            i_diy = idx_fd[0, 0]
            i_dfy = idx_fd[0, 2]

            value_dpressure_dx_gpu[i_dix:i_dfx, i_diy:i_dfy] = signal.convolve2d(
                pressure_gpu[i_dix:i_dfx, i_diy:i_dfy], x_kernel_gpu, mode='same')

            memory_dpressure_dx_gpu[i_dix:i_dfx, i_diy:i_dfy] = (
                    b_x_gpu[1:, :] * memory_dpressure_dx_gpu[i_dix:i_dfx, i_diy:i_dfy] +
                    a_x_gpu[1:, :] * value_dpressure_dx_gpu[i_dix:i_dfx, i_diy:i_dfy])

            value_dpressure_dx_gpu[i_dix:i_dfx, i_diy:i_dfy] = (
                    value_dpressure_dx_gpu[i_dix:i_dfx, i_diy:i_dfy] / k_x_gpu[1:, :] +
                    memory_dpressure_dx_gpu[i_dix:i_dfx, i_diy:i_dfy])

            vx_gpu = vx_gpu + self._dt * (value_dpressure_dx_gpu / rho_grid_vx_gpu)

            # segunda parte:  i: 1,NX-1; j: 1,NY-1 -> [1:-2, 1:-2]
            i_dix = idx_fd[0, 1]
            i_dfx = idx_fd[0, 3]
            i_diy = idx_fd[0, 1]
            i_dfy = idx_fd[0, 3]

            value_dpressure_dy_gpu[i_dix:i_dfx, i_diy:i_dfy] = signal.convolve2d(
                pressure_gpu[i_dix:i_dfx, i_diy + 1:i_dfy + 1], y_kernel_gpu, mode='same')

            memory_dpressure_dy_gpu[i_dix:i_dfx, i_diy:i_dfy] = (
                    b_y_half_gpu[:, :-1] * memory_dpressure_dy_gpu[i_dix:i_dfx, i_diy:i_dfy] +
                    a_y_half_gpu[:, :-1] * value_dpressure_dy_gpu[i_dix:i_dfx, i_diy:i_dfy])

            value_dpressure_dy_gpu[i_dix:i_dfx, i_diy:i_dfy] = \
                (value_dpressure_dy_gpu[i_dix:i_dfx, i_diy:i_dfy] / k_y_half_gpu[:, :-1] +
                memory_dpressure_dy_gpu[i_dix:i_dfx, i_diy:i_dfy])

            vy_gpu = vy_gpu + self._dt * (value_dpressure_dy_gpu / rho_grid_vy_gpu)
            
            # add the source (force vector located at a given grid point)
            for _isrc in range(self._n_pto_src):
                pressure_gpu[self._ix_src[_isrc], self._iy_src[_isrc]] += (self._source_term[it - 1, _isrc] * 
                                                                           self._dt * self._one_dx * self._one_dy)

            # implement Dirichlet boundary conditions on the six edges of the grid
            # which is the right condition to implement in order for C-PML to remain stable at long times
            # xmin
            vx_gpu[:ord - 1, :] = ZERO
            vy_gpu[:ord - 1 , :] = ZERO

            # xmax
            vx_gpu[-ord - 1:, :] = ZERO
            vy_gpu[-ord - 1:, :] = ZERO

            # ymin
            vx_gpu[:, :ord - 1] = ZERO
            vy_gpu[:, :ord - 1] = ZERO

            # ymax
            vx_gpu[:, -ord - 1:] = ZERO
            vy_gpu[:, -ord - 1:] = ZERO

            # Store seismograms
            for _i in range(self._idx_rec.shape[0]):
                _irec = self._idx_rec[_i]
                if it >= self._delay_recv[_irec]:
                    _x = self._ix_rec[_i]
                    _y = self._iy_rec[_i]
                    sens_vx[it - 1, _irec] += vx_gpu[_x, _y]
                    sens_vy[it - 1, _irec] += vy_gpu[_x, _y]
                    sens_pressure[it - 1, _irec] += pressure_gpu[_x, _y]

            psn2 = cupy.max(cupy.abs(pressure_gpu)).astype(flt32)
            if (it % self._it_display) == 0 or it == 5:
                if self._show_debug:
                    print(f'Time step # {it} out of {self._n_steps}')
                    print(f'Max pressure = {psn2}')

                if self._show_anim:
                    # self._windows_gpu[0].imv.setImage(vx_gpu[ix_min:ix_max, iy_min:iy_max].get(), levels=[v_min, v_max])
                    # self._windows_gpu[1].imv.setImage(vy_gpu[ix_min:ix_max, iy_min:iy_max].get(), levels=[v_min, v_max])
                    self._windows_gpu[0].imv.setImage(pressure_gpu[ix_min:ix_max, iy_min:iy_max].get(), levels=[v_min, v_max])
                    self._app.processEvents()

            # Verifica a estabilidade da simulacao
            if psn2 > STABILITY_THRESHOLD:
                raise StabilityError("Simulacao tornando-se instavel", psn2)
                
        sim_time = time() - t_gpu

        # Pega os resultados da simulacao
        vx = vx_gpu.get()
        vy = vy_gpu.get()
        pressure = pressure_gpu.get()

        # --------------------------------------------
        # A funcao de implementacao do simulador deve retornar
        # um dicionario com as seguintes chaves:
        #   - "vx": campo de velocidade no eixo x
        #   - "vygpu": campo de velocidade no eixo y
        #   - "pressuregpu": campo de pressao
        #   - "sens_vx": sinais de vx nos sensores
        #   - "sens_vy": sinais de vy nos sensores
        #   - "sens_pressure": sinais da pressao nos sensores
        #   - "gpu_str": string de identificacao da GPU utilizada na simulacao
        #   - "sim_time": tempo da simulacao, medido com a funcao time()
        # --------------------------------------------
        return {"vx": vx, "vy": vy, "pressure": pressure,
                "sens_vx": sens_vx, "sens_vy": sens_vy, "sens_pressure": sens_pressure,
                "gpu_str": cupy.cuda.runtime.getDeviceProperties(0)["name"].decode(), "sim_time": sim_time}
        

# ----------------------------------------------------------
# Avaliacao dos parametros na linha de comando
# ----------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', help='Configuration file', default='config.json')
args = parser.parse_args()

# Cria a instancia do simulador
sim_instance = SimulatorCupyConv(args.config)

# Executa simulacao
try:
    sim_instance.run()

except KeyError as key:
    print(f"Chave {key} nao encontrada no arquivo de configuracao.")
    
except ValueError as value:
    print(value)
