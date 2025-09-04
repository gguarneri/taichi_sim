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
class SimulatorCupyCuda(Simulator):
    def __init__(self, file_config):
        # Chama do construtor padrao, que le o arquivo de configuracao
        super().__init__(file_config)
        
        # Define o nome do simulador
        self._name = "CuPy-CUDA"
        
        
    def implementation(self):
        super().implementation()
        
        # --------------------------------------------
        # Aqui comeca o codigo especifico do simulador
        # --------------------------------------------
        # Transfere arrays de parametros para a GPU
        d_coefs = cupy.asarray(self._coefs)
        
        d_a_x = cupy.asarray(self._a_x)
        d_b_x = cupy.asarray(self._b_x)
        d_k_x = cupy.asarray(self._k_x)
        d_a_x_half = cupy.asarray(self._a_x_half)
        d_b_x_half = cupy.asarray(self._b_x_half)
        d_k_x_half = cupy.asarray(self._k_x_half)
        d_a_y = cupy.asarray(self._a_y)
        d_b_y = cupy.asarray(self._b_y)
        d_k_y = cupy.asarray(self._k_y)
        d_a_y_half = cupy.asarray(self._a_y_half)
        d_b_y_half = cupy.asarray(self._b_y_half)
        d_k_y_half = cupy.asarray(self._k_y_half)
        
        d_rho_grid_vx = cupy.asarray(self._rho_grid_vx)
        d_rho_grid_vy = cupy.asarray(self._rho_grid_vy)
        d_cp_grid_vx = cupy.asarray(self._cp_grid_vx)

        # Arrays para as variaveis de memoria do calculo  linearizado
        d_memory_dvx_dx = cupy.zeros((self._nx, self._ny), dtype=cupy.float32)
        d_memory_dvy_dy = cupy.zeros((self._nx, self._ny), dtype=cupy.float32)
        d_memory_dpressure_dx = cupy.zeros((self._nx, self._ny), dtype=cupy.float32)
        d_memory_dpressure_dy = cupy.zeros((self._nx, self._ny), dtype=cupy.float32)

        d_value_dvx_dx = cupy.zeros((self._nx, self._ny), dtype=cupy.float32)
        d_value_dvy_dy = cupy.zeros((self._nx, self._ny), dtype=cupy.float32)
        d_value_dpressure_dx = cupy.zeros((self._nx, self._ny), dtype=cupy.float32)
        d_value_dpressure_dy = cupy.zeros((self._nx, self._ny), dtype=cupy.float32)

        # Arrays dos campos de velocidade e pressoes - linearizado
        d_vx = cupy.zeros((self._nx, self._ny), dtype=cupy.float32)
        d_vy = cupy.zeros((self._nx, self._ny), dtype=cupy.float32)
        d_pressure = cupy.zeros((self._nx, self._ny), dtype=cupy.float32)
        d_pressure_l2_norm = cupy.asarray(np.zeros(1, dtype=flt32))
        
        # Arrays para os sensores
        d_sens_pressure = cupy.zeros((self._n_steps, self._n_rec), dtype=flt32)
        d_delay_rec = cupy.asarray(self._delay_recv)
        
        # Arrays com as informacoes sobre os elementos emissores (sources) e receptores (sensors)
        d_idx_src = cupy.asarray(self._pos_sources)
        d_idx_sen = cupy.asarray(self._pos_sensors)
        
        # Calculo dos indices para o staggered grid
        ord = self._coefs.shape[0]
        idx_fd = np.array([[c + 1, c, -c, -c - 1] for c in range(ord)], dtype=int32)
        d_idx_fd = cupy.asarray(idx_fd)

        # Definicao dos limites para a plotagem dos campos
        ix_min = self._roi.get_ix_min()
        ix_max = self._roi.get_ix_max()
        iy_min = self._roi.get_iz_min()
        iy_max = self._roi.get_iz_max()

        # Inicializa os mapas dos parametros de Lame
        d_kappa_unrelaxed = d_rho_grid_vx * d_cp_grid_vx * d_cp_grid_vx
        
        # Acrescenta eixo se source_term for array unidimensional
        if self._n_pto_src == 1:
            source_term = self._source_term[:, np.newaxis]
        d_source_term = cupy.asarray(source_term)
            
        # Cria os kernels
        with open('sim_cupy_cuda.cu') as kernel_file:
            kernel_string = kernel_file.read()
            pressure_kernel = cupy.RawKernel(kernel_string, 'pressure_kernel')
            velocity_kernel = cupy.RawKernel(kernel_string, 'velocity_kernel')

            self._block_size_x = np.gcd(self._nx, 16)
            self._block_size_y = np.gcd(self._ny, 16)
            block_size = (self._block_size_x, self._block_size_y)
            grid_velocity = ((self._nx + block_size[0] - 1) // block_size[0],
                            (self._ny + block_size[1] - 1) // block_size[1])

        # Laco de tempo para execucao da simulacao
        t_gpu = time()
        for it in range(1, self._n_steps + 1):
            # Calculo da pressao
            pressure_kernel(
                grid_velocity, block_size,
                (d_vx, d_vy, d_pressure, d_kappa_unrelaxed,
                d_memory_dvx_dx, d_memory_dvy_dy,
                d_value_dvx_dx, d_value_dvy_dy,
                d_a_x_half, d_b_x_half, d_k_x_half,
                d_a_y, d_b_y, d_k_y,
                d_coefs, d_idx_fd, d_idx_src, d_source_term, it, self._n_steps, self._n_pto_src,
                cupy.float32(self._dt), cupy.float32(1.0 / self._dx), cupy.float32(1.0 / self._dy),
                cupy.int32(self._nx), cupy.int32(self._ny), cupy.int32(ord), d_pressure_l2_norm)
            )
            
            # Calculo das velocidades
            velocity_kernel(
                grid_velocity, block_size,
                (d_vx, d_vy, d_pressure, d_rho_grid_vx, d_rho_grid_vy,
                d_memory_dpressure_dx, d_value_dpressure_dx,
                d_memory_dpressure_dy, d_value_dpressure_dy,
                d_a_x, d_b_x, d_k_x,
                d_a_y_half, d_b_y_half, d_k_y_half,
                d_coefs, d_idx_fd, d_sens_pressure, d_idx_sen, d_delay_rec, it, self._n_steps, self._n_rec,
                cupy.float32(self._dt), cupy.float32(1.0 / self._dx), cupy.float32(1.0 / self._dy),
                cupy.int32(self._nx), cupy.int32(self._ny), cupy.int32(ord), d_pressure_l2_norm)
            )

            psn2 = d_pressure_l2_norm.get()[0]
            if (it % self._it_display) == 0 or it == 5:
                if self._show_debug:
                    print(f"Time step {it} out of {self._n_steps}")
                    print(f"Max pressure = {psn2}")

                if self._show_anim:
                    self._windows_gpu[-1].imv.setImage(d_pressure[ix_min:ix_max, iy_min:iy_max].get(),
                                                       levels=[self._min_val_fields, self._max_val_fields])
                    self._app.processEvents()

            # Verifica a estabilidade da simulacao
            if psn2 > STABILITY_THRESHOLD:
                raise StabilityError("Simulacao tornando-se instavel", psn2)

        sim_time = time() - t_gpu

        # Pega os resultados da simulacao
        pressure = d_pressure.get()
        sens_pressure = d_sens_pressure.get()
        
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
                "gpu_str": cupy.cuda.runtime.getDeviceProperties(0)["name"].decode(), "sim_time": sim_time,
                "msg_impl": f'Block size: ({self._block_size_x}, {self._block_size_y})'}
        

# ----------------------------------------------------------
# Avaliacao dos parametros na linha de comando
# ----------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', help='Configuration file', default='config.json')
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
