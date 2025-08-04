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
import scipy

# -----------------------------------------------------------------------------
# Aqui deve ser implementado o simulador como uma classe herdada de Simulator
# -----------------------------------------------------------------------------
class SimulatorCpuConvUnsplit(Simulator):
    def __init__(self, file_config):
        # Chama do construtor padrao, que le o arquivo de configuracao
        super().__init__(file_config, sim_model="unsplit")
        
        # Define o nome do simulador
        self._name = "CPU-conv-unsplit"
        
        
    def implementation(self):
        super().implementation()
        
        # --------------------------------------------
        # Aqui comeca o codigo especifico do simulador
        # --------------------------------------------
        # Arrays para as variaveis de memoria do calculo
        memory_dpressure_dx = np.zeros((self._nx, self._ny), dtype=flt32)
        memory_dpressure_dy = np.zeros((self._nx, self._ny), dtype=flt32)
        memory_dpressurexx_dx = np.zeros((self._nx, self._ny), dtype=flt32)
        memory_dpressureyy_dy = np.zeros((self._nx, self._ny), dtype=flt32)

        dpressure_dx = np.zeros((self._nx, self._ny), dtype=flt32)
        dpressure_dy = np.zeros((self._nx, self._ny), dtype=flt32)
        dpressurexx_dx = np.zeros((self._nx, self._ny), dtype=flt32)
        dpressureyy_dy = np.zeros((self._nx, self._ny), dtype=flt32)

        # Arrays dos campos de velocidade e pressoes
        pressure_past = np.zeros((self._nx, self._ny), dtype=flt32)
        pressure_present = np.zeros((self._nx, self._ny), dtype=flt32)
        pressure_future = np.zeros((self._nx, self._ny), dtype=flt32)
        
        # Arrays para os sensores
        sens_pressure = np.zeros((self._n_steps, self._n_rec), dtype=flt32)

        # Calculo dos indices para o staggered grid
        ord = self._deriv_acc
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
        kappa = (self._rho_grid_vx * self._cp_grid_vx * self._cp_grid_vx)

        # Cria os kernels dos filtros para o calculo das derivadas parciais
        forward_x_kernel = np.array(self._coefs[::-1])[:, np.newaxis] * self._one_dx
        forward_y_kernel = np.array(self._coefs[::-1])[np.newaxis, :] * self._one_dy
        backward_x_kernel = np.array(-self._coefs)[:, np.newaxis] * self._one_dx
        backward_y_kernel = np.array(-self._coefs)[np.newaxis, :] * self._one_dy
        
        # Acrescenta eixo se source_term for array unidimensional
        if self._n_pto_src == 1:
            source_term = self._source_term[:, np.newaxis]

        # Inicio do laco de tempo
        t_gpu = time()
        for it in range(1, self._n_steps + 1):
            # Calculo das primeiras derivadas (forward) da pressao em relacao a x e y
            ia = idx_fd[last, 1]
            fa = idx_fd[last, 3]

            dpressure_dx[ia:fa, :] = scipy.signal.convolve(pressure_present, forward_x_kernel, mode='valid')
            dpressure_dy[:, ia:fa] = scipy.signal.convolve(pressure_present, forward_y_kernel, mode='valid')

            memory_dpressure_dx[ia:fa, :] = (self._b_x_half[:-1, :] * memory_dpressure_dx[ia:fa, :] +
                                             self._a_x_half[:-1, :] * dpressure_dx[ia:fa, :])
            memory_dpressure_dy[:, ia:fa] = (self._b_y_half[:, :-1] * memory_dpressure_dy[:, ia:fa] +
                                             self._a_y_half[:, :-1] * dpressure_dy[:, ia:fa])

            dpressure_dx[ia:fa, :] = dpressure_dx[ia:fa, :] / self._k_x_half[:-1, :] + memory_dpressure_dx[ia:fa, :]
            dpressure_dx /= self._rho_grid_vx
            dpressure_dy[:, ia:fa] = dpressure_dy[:, ia:fa] / self._k_y_half[:, :-1] + memory_dpressure_dy[:, ia:fa]
            dpressure_dy /= self._rho_grid_vy

            # Calculo das segundas derivada (backward) da pressao em relacao a x e y
            ia = idx_fd[last, 0]
            fa = idx_fd[last, 2]
            
            dpressurexx_dx[ia:fa, :] = scipy.signal.convolve(dpressure_dx, backward_x_kernel, mode='valid')
            dpressureyy_dy[:, ia:fa] = scipy.signal.convolve(dpressure_dy, backward_y_kernel, mode='valid')

            memory_dpressurexx_dx[ia:fa, :] = (self._b_x[1:, :] * memory_dpressurexx_dx[ia:fa, :] + 
                                               self._a_x[1:, :] * dpressurexx_dx[ia:fa, :])
            memory_dpressureyy_dy[:, ia:fa] = (self._b_y[:, 1:] * memory_dpressureyy_dy[:, ia:fa] + 
                                               self._a_y[:, 1:] * dpressureyy_dy[:, ia:fa])

            dpressurexx_dx[ia:fa, :] = dpressurexx_dx[ia:fa, :] / self._k_x[1:, :] + memory_dpressurexx_dx[ia:fa, :]
            dpressureyy_dy[:, ia:fa] = dpressureyy_dy[:, ia:fa] / self._k_y[:, 1:] + memory_dpressureyy_dy[:, ia:fa]
                        
            # Atualiza o campo de pressao futuro a partir do passado e do presente
            pressure_future = flt32(2.0) * pressure_present - pressure_past + self._dt**2 * (dpressurexx_dx + dpressureyy_dy) * kappa
            
            # Adicao das fontes no campo de pressao
            for _isrc in range(self._n_pto_src):
                pressure_future[self._ix_src[_isrc], self._iy_src[_isrc]] += (source_term[it - 1, _isrc] * self._dt**2 *
                                                                              self._one_dx * self._one_dy)
           
            # Aplica as condicoes de Dirichlet
            # xmin
            pressure_future[0, :] = ZERO

            # xmax
            pressure_future[-1, :] = ZERO

            # ymin
            pressure_future[:, 0] = ZERO

            # ymax
            pressure_future[:, -1] = ZERO

            # Armazena os sinais dos sensores
            for _i in range(self._idx_rec.shape[0]):
                _irec = self._idx_rec[_i]
                if it >= self._delay_recv[_irec]:
                    _x = self._ix_rec[_i]
                    _y = self._iy_rec[_i]
                    sens_pressure[it - 1, _irec] += pressure_future[_x, _y]

            psn2 = np.max(np.abs(pressure_future)).astype(flt32)
            if (it % self._it_display) == 0 or it == 5:
                if self._show_debug:
                    print(f"Time step {it} out of {self._n_steps}")
                    print(f"Max absolute value of pressure = {psn2}")

                if self._show_anim:
                    self._windows_gpu[0].imv.setImage(pressure_future[ix_min:ix_max, iy_min:iy_max], levels=[v_min, v_max])
                    self._app.processEvents()
                    
            # Swap dos valores novos de pressao para valores antigos
            pressure_past = pressure_present
            pressure_present = pressure_future

            # Verifica a estabilidade da simulacao
            if psn2 > STABILITY_THRESHOLD:
                raise StabilityError("Simulacao tornando-se instavel", psn2)
                
        sim_time = time() - t_gpu

        # --------------------------------------------
        # A funcao de implementacao do simulador deve retornar
        # um dicionario com as seguintes chaves:
        #   - "pressure": campo de pressao
        #   - "sens_pressure": sinais da pressao nos sensores
        #   - "gpu_str": string de identificacao da GPU utilizada na simulacao
        #   - "sim_time": tempo da simulacao, medido com a funcao time()
        #   - opcionalmente pode ter uma mensagem exclusiva da implementacao em "msg_impl"
        # --------------------------------------------
        return {"pressure": pressure_present, "sens_pressure": sens_pressure,
                "gpu_str": "CPU - conv", "sim_time": sim_time}
        

# ----------------------------------------------------------
# Avaliacao dos parametros na linha de comando
# ----------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', help='Configuration file', default='config.json')
args = parser.parse_args()

# Cria a instancia do simulador
sim_instance = SimulatorCpuConvUnsplit(args.config)

# Executa simulacao
try:
    sim_instance.run()

except KeyError as key:
    print(f"Chave {key} nao encontrada no arquivo de configuracao.")
    
except ValueError as value:
    print(value)
