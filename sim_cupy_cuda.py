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
        d_idx_fd = cupy.asarray(idx_fd)

        # Definicao dos limites para a plotagem dos campos
        v_max = 100.0
        v_min = -v_max
        ix_min = self._roi.get_ix_min()
        ix_max = self._roi.get_ix_max()
        iy_min = self._roi.get_iz_min()
        iy_max = self._roi.get_iz_max()

        # Inicializa os mapas dos parametros de Lame
        d_kappa_unrelaxed = d_rho_grid_vx * d_cp_grid_vx * d_cp_grid_vx
        
        # Acrescenta eixo se source_term for array unidimensional
        if self._n_pto_src == 1:
            source_term = self._source_term[:, np.newaxis]
            
        # Cria os kernels
        with open('sim_cupy_cuda.cu') as kernel_file:
            kernel_string = kernel_file.read()
            pressure_kernel = cupy.RawKernel(kernel_string, 'pressure_kernel')
            velocity_vx_kernel = cupy.RawKernel(kernel_string, 'velocity_vx_kernel')
            velocity_vy_kernel = cupy.RawKernel(kernel_string, 'velocity_vy_kernel')
            dirichlet_boundary_kernel = cupy.RawKernel(kernel_string, 'dirichlet_boundary_kernel')
            test_kernel = cupy.RawKernel(kernel_string, 'test_kernel')

            self._block_size_x = np.gcd(self._nx, 16)
            self._block_size_y = np.gcd(self._ny, 16)
            block_size = (self._block_size_x, self._block_size_y)
            grid_velocity = ((self._nx + block_size[0] - 1) // block_size[0],
                            (self._ny + block_size[1] - 1) // block_size[1])
            grid_boundary = ((self._nx + block_size[0] - 1) // block_size[0],
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
                d_coefs, d_idx_fd,
                cupy.float32(self._dt), cupy.float32(1.0 / self._dx), cupy.float32(1.0 / self._dy),
                cupy.int32(self._nx), cupy.int32(self._ny), cupy.int32(ord))
            )
            
            # Adicao das fontes no campo de pressao
            for _isrc in range(self._n_pto_src):
                d_pressure[self._ix_src[_isrc], self._iy_src[_isrc]] += (source_term[it - 1, _isrc] *
                                                                         self._dt * self._one_dx * self._one_dy)

            # Calculo da velocidade vx
            velocity_vx_kernel(
                grid_velocity, block_size,
                (d_vx, d_pressure, d_rho_grid_vx,
                d_memory_dpressure_dx, d_value_dpressure_dx,
                d_a_x, d_b_x, d_k_x,
                d_coefs, d_idx_fd,
                cupy.float32(self._dt), cupy.float32(1.0 / self._dx), cupy.float32(1.0 / self._dy),
                cupy.int32(self._nx), cupy.int32(self._ny), cupy.int32(ord))
            )

            # Calculo da velocidade vy
            velocity_vy_kernel(
                grid_velocity, block_size,
                (d_vy, d_pressure, d_rho_grid_vy,
                d_memory_dpressure_dy, d_value_dpressure_dy,
                d_a_y_half, d_b_y_half, d_k_y_half,
                d_coefs, d_idx_fd,
                cupy.float32(self._dt), cupy.float32(1.0 / self._dx), cupy.float32(1.0 / self._dy),
                cupy.int32(self._nx), cupy.int32(self._ny), cupy.int32(ord))
            )

            # Aplica as condicoes de Dirichlet
            dirichlet_boundary_kernel(
                grid_boundary, block_size,
                (d_vx, d_vy, cupy.int32(self._nx), cupy.int32(self._ny), cupy.int32(ord))
            )

            # Armazena os sinais dos sensores
            for _i in range(self._idx_rec.shape[0]):
                _irec = self._idx_rec[_i]
                if it >= self._delay_recv[_irec]:
                    _x = self._ix_rec[_i]
                    _y = self._iy_rec[_i]
                    sens_vx[it - 1, _irec] += d_vx[_x, _y]
                    sens_vy[it - 1, _irec] += d_vy[_x, _y]
                    sens_pressure[it - 1, _irec] += d_pressure[_x, _y]

            psn2 = cupy.max(cupy.abs(d_pressure)).astype(flt32)
            if (it % self._it_display) == 0 or it == 5:
                if self._show_debug:
                    print(f"Time step {it} out of {self._n_steps}")
                    print(f"Max pressure = {psn2}")

                if self._show_anim:
                    self._windows_gpu[-1].imv.setImage(d_pressure[ix_min:ix_max, iy_min:iy_max].get(), levels=[v_min, v_max])
                    self._app.processEvents()

            # Verifica a estabilidade da simulacao
            if psn2 > STABILITY_THRESHOLD:
                raise StabilityError("Simulacao tornando-se instavel", psn2)

        sim_time = time() - t_gpu

        # Pega os resultados da simulacao
        pressure = d_pressure.get()
        
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
