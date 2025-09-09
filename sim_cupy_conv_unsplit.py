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
class SimulatorCupyConvUnsplit(Simulator):
    def __init__(self, file_config):
        # Chama do construtor padrao, que le o arquivo de configuracao
        super().__init__(file_config, sim_model="unsplit")
        
        # Define o nome do simulador
        self._name = "CuPy-conv-unsplit"
        
        
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
        memory_dpressure_dx_gpu = cupy.asarray(np.zeros((self._nx, self._ny), dtype=flt32))
        memory_dpressure_dy_gpu = cupy.asarray(np.zeros((self._nx, self._ny), dtype=flt32))
        memory_dpressurexx_dx_gpu = cupy.asarray(np.zeros((self._nx, self._ny), dtype=flt32))
        memory_dpressureyy_dy_gpu = cupy.asarray(np.zeros((self._nx, self._ny), dtype=flt32))

        dpressure_dx_gpu = cupy.asarray(np.zeros((self._nx, self._ny), dtype=flt32))
        dpressure_dy_gpu = cupy.asarray(np.zeros((self._nx, self._ny), dtype=flt32))
        dpressurexx_dx_gpu = cupy.asarray(np.zeros((self._nx, self._ny), dtype=flt32))
        dpressureyy_dy_gpu = cupy.asarray(np.zeros((self._nx, self._ny), dtype=flt32))
        
        # Arrays dos campos de velocidade e pressoes
        pressure_past_gpu = cupy.asarray(np.zeros((self._nx, self._ny), dtype=flt32))
        pressure_present_gpu = cupy.asarray(np.zeros((self._nx, self._ny), dtype=flt32))
        pressure_future_gpu = cupy.asarray(np.zeros((self._nx, self._ny), dtype=flt32))
        
        # Arrays para os sensores
        sens_pressure = np.zeros((self._n_steps, self._n_rec), dtype=flt32)

        # Calculo dos indices para o staggered grid
        ord = self._deriv_acc
        idx_fd = np.array([[c + 1, c, -c, -c - 1] for c in range(ord)], dtype=int32)
        last = ord - 1

        # Definicao dos limites para a plotagem dos campos
        ix_min = self._roi.get_ix_min()
        ix_max = self._roi.get_ix_max()
        iy_min = self._roi.get_iz_min()
        iy_max = self._roi.get_iz_max()

        # Inicializa os mapas dos parametros de Lame
        kappa_gpu = cupy.asarray(self._rho_grid_vx * self._cp_grid_vx * self._cp_grid_vx)

        # Cria os kernels dos filtros para o calculo das derivadas parciais
        forward_x_kernel_gpu = cupy.asarray(np.array(self._coefs[::-1])[:, np.newaxis] * self._one_dx)
        forward_y_kernel_gpu = cupy.asarray(np.array(self._coefs[::-1])[np.newaxis, :] * self._one_dy)
        backward_x_kernel_gpu = cupy.asarray(np.array(-self._coefs)[:, np.newaxis] * self._one_dx)
        backward_y_kernel_gpu = cupy.asarray(np.array(-self._coefs)[np.newaxis, :] * self._one_dy)

        # Acrescenta eixo se source_term for array unidimensional
        if self._n_pto_src == 1:
            source_term = self._source_term[:, np.newaxis]
            
        # Laco de tempo para execucao da simulacao
        t_gpu = time()
        for it in range(1, self._n_steps + 1):
            # Calculo das primeiras derivadas (forward) da pressao em relacao a x e y
            ia = idx_fd[last, 1]
            fa = idx_fd[last, 3]

            dpressure_dx_gpu[ia:fa, :] = signal.convolve2d(pressure_present_gpu, forward_x_kernel_gpu, mode='valid')
            dpressure_dy_gpu[:, ia:fa] = signal.convolve2d(pressure_present_gpu, forward_y_kernel_gpu, mode='valid')

            memory_dpressure_dx_gpu[ia:fa, :] = (b_x_half_gpu[:-1, :] * memory_dpressure_dx_gpu[ia:fa, :] +
                                                 a_x_half_gpu[:-1, :] * dpressure_dx_gpu[ia:fa, :])
            memory_dpressure_dy_gpu[:, ia:fa] = (b_y_half_gpu[:, :-1] * memory_dpressure_dy_gpu[:, ia:fa] +
                                                 a_y_half_gpu[:, :-1] * dpressure_dy_gpu[:, ia:fa])

            dpressure_dx_gpu[ia:fa, :] = dpressure_dx_gpu[ia:fa, :] / k_x_half_gpu[:-1, :] + memory_dpressure_dx_gpu[ia:fa, :]
            dpressure_dx_gpu /= rho_grid_vx_gpu
            dpressure_dy_gpu[:, ia:fa] = dpressure_dy_gpu[:, ia:fa] / k_y_half_gpu[:, :-1] + memory_dpressure_dy_gpu[:, ia:fa]
            dpressure_dy_gpu /= rho_grid_vy_gpu
            
            # Calculo das segundas derivada (backward) da pressao em relacao a x e y
            ia = idx_fd[last, 0]
            fa = idx_fd[last, 2]
            
            dpressurexx_dx_gpu[ia:fa, :] = signal.convolve2d(dpressure_dx_gpu, backward_x_kernel_gpu, mode='valid')
            dpressureyy_dy_gpu[:, ia:fa] = signal.convolve2d(dpressure_dy_gpu, backward_y_kernel_gpu, mode='valid')

            memory_dpressurexx_dx_gpu[ia:fa, :] = (b_x_gpu[1:, :] * memory_dpressurexx_dx_gpu[ia:fa, :] + 
                                                   a_x_gpu[1:, :] * dpressurexx_dx_gpu[ia:fa, :])
            memory_dpressureyy_dy_gpu[:, ia:fa] = (b_y_gpu[:, 1:] * memory_dpressureyy_dy_gpu[:, ia:fa] + 
                                                   a_y_gpu[:, 1:] * dpressureyy_dy_gpu[:, ia:fa])

            dpressurexx_dx_gpu[ia:fa, :] = dpressurexx_dx_gpu[ia:fa, :] / k_x_gpu[1:, :] + memory_dpressurexx_dx_gpu[ia:fa, :]
            dpressureyy_dy_gpu[:, ia:fa] = dpressureyy_dy_gpu[:, ia:fa] / k_y_gpu[:, 1:] + memory_dpressureyy_dy_gpu[:, ia:fa]
                        
            # Atualiza o campo de pressao futuro a partir do passado e do presente
            pressure_future_gpu = (flt32(2.0) * pressure_present_gpu - pressure_past_gpu +
                                   self._dt**2 * (dpressurexx_dx_gpu + dpressureyy_dy_gpu) * kappa_gpu)

            # Adicao das fontes no campo de pressao
            for _isrc in range(self._n_pto_src):
                pressure_future_gpu[self._ix_src[_isrc], self._iy_src[_isrc]] += (source_term[it - 1, _isrc] * self._dt**2 *
                                                                                  self._one_dx * self._one_dy)

            # Aplica as condicoes de Dirichlet
            # xmin
            pressure_future_gpu[0, :] = ZERO

            # xmax
            pressure_future_gpu[-1, :] = ZERO

            # ymin
            pressure_future_gpu[:, 0] = ZERO

            # ymax
            pressure_future_gpu[:, -1] = ZERO

            # Armazena os sinais dos sensores
            for _i in range(self._idx_rec.shape[0]):
                _irec = self._idx_rec[_i]
                if it >= self._delay_recv[_irec]:
                    _x = self._ix_rec[_i]
                    _y = self._iy_rec[_i]
                    sens_pressure[it - 1, _irec] += pressure_future_gpu[_x, _y]

            psn2 = cupy.max(cupy.abs(pressure_future_gpu)).astype(flt32)
            if (it % self._it_display) == 0 or it == 5:
                if self._show_debug:
                    print(f"Time step {it} out of {self._n_steps}")
                    print(f"Max absolute value of pressure = {psn2}")

                if self._show_anim:
                    self._windows_gpu[0].imv.setImage(pressure_future_gpu[ix_min:ix_max, iy_min:iy_max].get(),
                                                      levels=[self._min_val_fields, self._max_val_fields])
                    self._app.processEvents()
                    
            # Swap dos valores novos de pressao para valores antigos
            pressure_past_gpu = pressure_present_gpu
            pressure_present_gpu = pressure_future_gpu

            # Verifica a estabilidade da simulacao
            if psn2 > STABILITY_THRESHOLD:
                raise StabilityError("Simulacao tornando-se instavel", psn2)
            
        sim_time = time() - t_gpu

        # Pega os resultados da simulacao
        pressure = pressure_future_gpu.get()
        
        # Libera a memoria alocada na GPU
        cupy.get_default_memory_pool().free_all_blocks()
        cupy.get_default_pinned_memory_pool().free_all_blocks()

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
                "gpu_str": cupy.cuda.runtime.getDeviceProperties(0)["name"].decode(), "sim_time": sim_time}
        

# ----------------------------------------------------------
# Avaliacao dos parametros na linha de comando
# ----------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', help='Configuration file', default='config.json')
args = parser.parse_args()

# Cria a instancia do simulador
sim_instance = SimulatorCupyConvUnsplit(args.config)

# Executa simulacao
try:
    sim_instance.run()

except KeyError as key:
    print(f"Chave {key} nao encontrada no arquivo de configuracao.")
    
except ValueError as value:
    print(value)
