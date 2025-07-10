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


# -----------------------------------------------------------------------------
# Aqui deve ser implementado o simulador como uma classe herdada de Simulator
# -----------------------------------------------------------------------------
class SimulatorCpuBroadcast(Simulator):
    def __init__(self, file_config):
        # Chama do construtor padrao, que le o arquivo de configuracao
        super().__init__(file_config)
        
        # Define o nome do simulador
        self._name = "CPU-broadcast"
        
        
    def implementation(self):
        super().implementation()
        
        # --------------------------------------------
        # Aqui comeca o codigo especifico do simulador
        # --------------------------------------------
        # Arrays para as variaveis de memoria do calculo
        memory_dvx_dx = np.zeros((self._nx, self._ny), dtype=flt32)
        memory_dvy_dy = np.zeros((self._nx, self._ny), dtype=flt32)
        memory_dpressure_dx = np.zeros((self._nx, self._ny), dtype=flt32)
        memory_dpressure_dy = np.zeros((self._nx, self._ny), dtype=flt32)

        value_dvx_dx = np.zeros((self._nx, self._ny), dtype=flt32)
        value_dvy_dy = np.zeros((self._nx, self._ny), dtype=flt32)
        value_dpressure_dx = np.zeros((self._nx, self._ny), dtype=flt32)
        value_dpressure_dy = np.zeros((self._nx, self._ny), dtype=flt32)

        # Arrays dos campos de velocidade e pressoes
        vx = np.zeros((self._nx, self._ny), dtype=flt32)
        vy = np.zeros((self._nx, self._ny), dtype=flt32)
        pressure = np.zeros((self._nx, self._ny), dtype=flt32)
        
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
        kappa_unrelaxed = self._rho_grid_vx * self._cp_grid_vx * self._cp_grid_vx
        
        # Acrescenta eixo se source_term for array unidimensional
        if self._n_pto_src == 1:
            self._source_term = self._source_term[:, np.newaxis]

        # Inicio do laco de tempo
        t_gpu = time()
        for it in range(1, self._n_steps + 1):
            # Calculo da pressao
            # Primeiro "laco" i: 1,NX-1; j: 2,NY -> [1:-2, 2:-1]
            i_dix = idx_fd[0, 1]
            i_dfx = idx_fd[0, 3]
            i_diy = idx_fd[0, 0]
            i_dfy = idx_fd[0, 2]
            for c in range(ord):
                # Eixo "x"
                i_iax = None if idx_fd[c, 0] == 0 else idx_fd[c, 0]
                i_fax = None if idx_fd[c, 2] == 0 else idx_fd[c, 2]
                i_ibx = None if idx_fd[c, 1] == 0 else idx_fd[c, 1]
                i_fbx = None if idx_fd[c, 3] == 0 else idx_fd[c, 3]
                # eixo "y"
                i_iay = None if idx_fd[c, 0] == 0 else idx_fd[c, 0]
                i_fay = None if idx_fd[c, 2] == 0 else idx_fd[c, 2]
                i_iby = None if idx_fd[c, 1] == 0 else idx_fd[c, 1]
                i_fby = None if idx_fd[c, 3] == 0 else idx_fd[c, 3]
                if c:
                    value_dvx_dx[i_dix:i_dfx, i_diy:i_dfy] += \
                        (self._coefs[c] * (vx[i_iax:i_fax, i_diy:i_dfy] - vx[i_ibx:i_fbx, i_diy:i_dfy]) * self._one_dx)
                    value_dvy_dy[i_dix:i_dfx, i_diy:i_dfy] += \
                        (self._coefs[c] * (vy[i_dix:i_dfx, i_iay:i_fay] - vy[i_dix:i_dfx, i_iby:i_fby]) * self._one_dy)
                else:
                    value_dvx_dx[i_dix:i_dfx, i_diy:i_dfy] = \
                        (self._coefs[c] * (vx[i_iax:i_fax, i_diy:i_dfy] - vx[i_ibx:i_fbx, i_diy:i_dfy]) * self._one_dx)
                    value_dvy_dy[i_dix:i_dfx, i_diy:i_dfy] = \
                        (self._coefs[c] * (vy[i_dix:i_dfx, i_iay:i_fay] - vy[i_dix:i_dfx, i_iby:i_fby]) * self._one_dy)

            memory_dvx_dx[i_dix:i_dfx, i_diy:i_dfy] = (self._b_x_half[:-1, :] * memory_dvx_dx[i_dix:i_dfx, i_diy:i_dfy] +
                                                       self._a_x_half[:-1, :] * value_dvx_dx[i_dix:i_dfx, i_diy:i_dfy])
            memory_dvy_dy[i_dix:i_dfx, i_diy:i_dfy] = (self._b_y[:, 1:] * memory_dvy_dy[i_dix:i_dfx, i_diy:i_dfy] +
                                                       self._a_y[:, 1:] * value_dvy_dy[i_dix:i_dfx, i_diy:i_dfy])
        
            value_dvx_dx[i_dix:i_dfx, i_diy:i_dfy] = (value_dvx_dx[i_dix:i_dfx, i_diy:i_dfy] / self._k_x_half[:-1, :] +
                                                    memory_dvx_dx[i_dix:i_dfx, i_diy:i_dfy])
            value_dvy_dy[i_dix:i_dfx, i_diy:i_dfy] = (value_dvy_dy[i_dix:i_dfx, i_diy:i_dfy] / self._k_y[:, 1:] +
                                                    memory_dvy_dy[i_dix:i_dfx, i_diy:i_dfy])
            
            # compute the pressure using the Lame parameters
            pressure += kappa_unrelaxed * (value_dvx_dx + value_dvy_dy) * self._dt * self._one_dx * self._one_dy

            # add the source (force vector located at a given grid point)
            # add the source (force vector located at a given grid point)
            for _isrc in range(self._n_pto_src):
                pressure[self._ix_src[_isrc], self._iy_src[_isrc]] += (self._source_term[it - 1, _isrc] *
                                                                       self._dt * self._one_dx * self._one_dy)

            # Calculo da velocidade
            # Primeiro "laco" i: 2,NX; j: 2,NY -> [2:-1, 2:-1]
            i_dix = idx_fd[0, 0]
            i_dfx = idx_fd[0, 2]
            i_diy = idx_fd[0, 0]
            i_dfy = idx_fd[0, 2]
            for c in range(ord):
                # Eixo "x"
                i_iax = None if idx_fd[c, 0] == 0 else idx_fd[c, 0]
                i_fax = None if idx_fd[c, 2] == 0 else idx_fd[c, 2]
                i_ibx = None if idx_fd[c, 1] == 0 else idx_fd[c, 1]
                i_fbx = None if idx_fd[c, 3] == 0 else idx_fd[c, 3]
                if c:
                    value_dpressure_dx[i_dix:i_dfx, i_diy:i_dfy] += \
                        (self._coefs[c] * (pressure[i_iax:i_fax, i_diy:i_dfy] - pressure[i_ibx:i_fbx, i_diy:i_dfy]) * self._one_dx)
                else:
                    value_dpressure_dx[i_dix:i_dfx, i_diy:i_dfy] = \
                        (self._coefs[c] * (pressure[i_iax:i_fax, i_diy:i_dfy] - pressure[i_ibx:i_fbx, i_diy:i_dfy]) * self._one_dx)

            memory_dpressure_dx[i_dix:i_dfx, i_diy:i_dfy] = (self._b_x[1:, :] * memory_dpressure_dx[i_dix:i_dfx, i_diy:i_dfy] +
                                                             self._a_x[1:, :] * value_dpressure_dx[i_dix:i_dfx, i_diy:i_dfy])

            value_dpressure_dx[i_dix:i_dfx, i_diy:i_dfy] = (value_dpressure_dx[i_dix:i_dfx, i_diy:i_dfy] / self._k_x[1:, :] +
                                                            memory_dpressure_dx[i_dix:i_dfx, i_diy:i_dfy])

            vx += self._dt * (value_dpressure_dx / self._rho_grid_vx)

            # segunda parte:  i: 1,NX-1; j: 1,NY-1 -> [1:-2, 1:-2]
            i_dix = idx_fd[0, 1]
            i_dfx = idx_fd[0, 3]
            i_diy = idx_fd[0, 1]
            i_dfy = idx_fd[0, 3]
            for c in range(ord):
                # eixo "y"
                i_iay = None if idx_fd[c, 0] == 0 else idx_fd[c, 0]
                i_fay = None if idx_fd[c, 2] == 0 else idx_fd[c, 2]
                i_iby = None if idx_fd[c, 1] == 0 else idx_fd[c, 1]
                i_fby = None if idx_fd[c, 3] == 0 else idx_fd[c, 3]
                if c:
                    value_dpressure_dy[i_dix:i_dfx, i_diy:i_dfy] += (
                            self._coefs[c] * (pressure[i_dix:i_dfx, i_iay:i_fay] - pressure[i_dix:i_dfx, i_iby:i_fby]) * self._one_dy)
                else:
                    value_dpressure_dy[i_dix:i_dfx, i_diy:i_dfy] = (
                            self._coefs[c] * (pressure[i_dix:i_dfx, i_iay:i_fay] - pressure[i_dix:i_dfx, i_iby:i_fby]) * self._one_dy)

            memory_dpressure_dy[i_dix:i_dfx, i_diy:i_dfy] = (
                    self._b_y_half[:, :-1] * memory_dpressure_dy[i_dix:i_dfx, i_diy:i_dfy] +
                    self._a_y_half[:, :-1] * value_dpressure_dy[i_dix:i_dfx, i_diy:i_dfy])

            value_dpressure_dy[i_dix:i_dfx, i_diy:i_dfy] = \
                (value_dpressure_dy[i_dix:i_dfx, i_diy:i_dfy] / self._k_y_half[:, :-1] +
                memory_dpressure_dy[i_dix:i_dfx, i_diy:i_dfy])

            vy += self._dt * (value_dpressure_dy / self._rho_grid_vy)

            # implement Dirichlet boundary conditions on the six edges of the grid
            # which is the right condition to implement in order for C-PML to remain stable at long times
            # xmin
            vx[:(ord - 1), :] = ZERO
            vy[:(ord - 1), :] = ZERO

            # xmax
            vx[-(ord - 1):, :] = ZERO
            vy[-(ord - 1):, :] = ZERO

            # ymin
            vx[:, :(ord - 1)] = ZERO
            vy[:, :(ord - 1)] = ZERO

            # ymax
            vx[:, -(ord - 1):] = ZERO
            vy[:, -(ord - 1):] = ZERO

            # Store seismograms
            for _i in range(self._idx_rec.shape[0]):
                _irec = self._idx_rec[_i]
                if it >= self._delay_recv[_irec]:
                    _x = self._ix_rec[_i]
                    _y = self._iy_rec[_i]
                    sens_vx[it - 1, _irec] += vx[_x, _y]
                    sens_vy[it - 1, _irec] += vy[_x, _y]
                    sens_pressure[it - 1, _irec] += pressure[_x, _y]

            psn2 = np.max(np.abs(pressure)).astype(flt32)
            if (it % self._it_display) == 0 or it == 5:
                if self._show_debug:
                    print(f"Time step {it} out of {self._n_steps}")
                    print(f"Max absolute value of pressure = {psn2}")

                if self._show_anim:
                    # self._windows_gpu[0].imv.setImage(vx[ix_min:ix_max, iy_min:iy_max], levels=[v_min, v_max])
                    # self._windows_gpu[1].imv.setImage(vy[ix_min:ix_max, iy_min:iy_max], levels=[v_min, v_max])
                    self._windows_gpu[0].imv.setImage(pressure[ix_min:ix_max, iy_min:iy_max], levels=[v_min, v_max])
                    self._app.processEvents()

            # Verifica a estabilidade da simulacao
            if psn2 > STABILITY_THRESHOLD:
                raise StabilityError("Simulacao tornando-se instavel", psn2)
            
        sim_time = time() - t_gpu

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
                "gpu_str": "CPU - broadcast", "sim_time": sim_time}
        

# ----------------------------------------------------------
# Avaliacao dos parametros na linha de comando
# ----------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', help='Configuration file', default='config.json')
args = parser.parse_args()

# Cria a instancia do simulador
sim_instance = SimulatorCpuBroadcast(args.config)

# Executa simulacao
try:
    sim_instance.run()

except KeyError as key:
    print(f"Chave {key} nao encontrada no arquivo de configuracao.")
    
except ValueError as value:
    print(value)
