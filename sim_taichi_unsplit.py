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
import taichi as ti


# -----------------------------------------------------------------------------
# Aqui deve ser implementado o simulador como uma classe herdada de Simulator
# -----------------------------------------------------------------------------
class SimulatorTaichiUnsplit(Simulator):
    def __init__(self, file_config):
        # Chama do construtor padrao, que le o arquivo de configuracao
        super().__init__(file_config, sim_model="unsplit")
        
        # Define o nome do simulador
        self._name = "Taichi-unsplit"
        
        
    def implementation(self):
        # ---------------------------------
        # Implementacao do simulador
        # ---------------------------------
        super().implementation()
        
        # --------------------------------------------
        # Aqui comeca o codigo especifico do simulador
        # --------------------------------------------
        # Inicializacao do framework
        ti.init(arch=ti.gpu, debug=False, print_ir=False, kernel_profiler=False)
        
        # Transfere arrays de parametros para a GPU
        a_x_gpu = ti.field(dtype=float, shape=self._a_x.flatten().shape)
        a_x_gpu.from_numpy(self._a_x.flatten())
        b_x_gpu = ti.field(dtype=float, shape=self._b_x.flatten().shape)
        b_x_gpu.from_numpy(self._b_x.flatten())
        k_x_gpu = ti.field(dtype=float, shape=self._k_x.flatten().shape)
        k_x_gpu.from_numpy(self._k_x.flatten())
        a_x_half_gpu = ti.field(dtype=float, shape=self._a_x_half.flatten().shape)
        a_x_half_gpu.from_numpy(self._a_x_half.flatten())
        b_x_half_gpu = ti.field(dtype=float, shape=self._b_x_half.flatten().shape)
        b_x_half_gpu.from_numpy(self._b_x_half.flatten())
        k_x_half_gpu = ti.field(dtype=float, shape=self._k_x_half.flatten().shape)
        k_x_half_gpu.from_numpy(self._k_x_half.flatten())
        
        a_y_gpu = ti.field(dtype=float, shape=self._a_y.flatten().shape)
        a_y_gpu.from_numpy(self._a_y.flatten())
        b_y_gpu = ti.field(dtype=float, shape=self._b_y.flatten().shape)
        b_y_gpu.from_numpy(self._b_y.flatten())
        k_y_gpu = ti.field(dtype=float, shape=self._k_y.flatten().shape)
        k_y_gpu.from_numpy(self._k_y.flatten())
        a_y_half_gpu = ti.field(dtype=float, shape=self._a_y_half.flatten().shape)
        a_y_half_gpu.from_numpy(self._a_y_half.flatten())
        b_y_half_gpu = ti.field(dtype=float, shape=self._b_y_half.flatten().shape)
        b_y_half_gpu.from_numpy(self._b_y_half.flatten())
        k_y_half_gpu = ti.field(dtype=float, shape=self._k_y_half.flatten().shape)
        k_y_half_gpu.from_numpy(self._k_y_half.flatten())
        
        rho_grid_vx_gpu = ti.field(dtype=float, shape=self._rho_grid_vx.shape)
        rho_grid_vx_gpu.from_numpy(self._rho_grid_vx)
        rho_grid_vy_gpu = ti.field(dtype=float, shape=self._rho_grid_vy.shape)
        rho_grid_vy_gpu.from_numpy(self._rho_grid_vy)
        coefs_gpu = ti.field(dtype=float, shape=self._coefs.shape)
        coefs_gpu.from_numpy(self._coefs)
        
        # Arrays para as variaveis de memoria do calculo
        memory_dpressure_dx_gpu = ti.field(dtype=float, shape=(self._nx, self._ny))
        memory_dpressure_dy_gpu = ti.field(dtype=float, shape=(self._nx, self._ny))
        memory_dpressurexx_dx_gpu = ti.field(dtype=float, shape=(self._nx, self._ny))
        memory_dpressureyy_dy_gpu = ti.field(dtype=float, shape=(self._nx, self._ny))

        dpressure_dx_gpu = ti.field(dtype=float, shape=(self._nx, self._ny))
        dpressure_dy_gpu = ti.field(dtype=float, shape=(self._nx, self._ny))

        # Arrays dos campos de pressao
        pressure_past_gpu = ti.field(dtype=float, shape=(self._nx, self._ny))
        pressure_present_gpu = ti.field(dtype=float, shape=(self._nx, self._ny))
        pressure_future_gpu = ti.field(dtype=float, shape=(self._nx, self._ny))
        pressure_l2_norm_gpu = ti.field(dtype=float, shape=())
        
        # Arrays para os sensores
        sens_pressure_gpu = ti.field(dtype=float, shape=(self._n_steps, self._n_rec))
        delay_rec_gpu = ti.field(dtype=int, shape=self._delay_recv.shape)
        delay_rec_gpu.from_numpy(self._delay_recv)

        # Calculo dos indices para o staggered grid
        ord = self._deriv_acc
        idx_fd = np.array([[c + 1, c, -c, -c - 1] for c in range(ord)], dtype=int32)
        idx_fd_gpu = ti.field(dtype=int, shape=idx_fd.shape)
        idx_fd_gpu.from_numpy(idx_fd)

        # Definicao dos limites para a plotagem dos campos
        ix_min = self._roi.get_ix_min()
        ix_max = self._roi.get_ix_max()
        iy_min = self._roi.get_iz_min()
        iy_max = self._roi.get_iz_max()

        # Inicializa os mapas dos parametros de Lame
        kappa_gpu = ti.field(dtype=float, shape=(self._nx, self._ny))
        kappa_gpu.from_numpy(self._rho_grid_vx * self._cp_grid_vx * self._cp_grid_vx)

        # Acrescenta eixo se source_term for array unidimensional
        if self._n_pto_src == 1:
            source_term = self._source_term[:, np.newaxis]
        source_term_gpu = ti.field(dtype=float, shape=source_term.shape)
        source_term_gpu.from_numpy(source_term)
        
        # Arrays com as informacoes sobre os elementos emissores (sources) e receptores (sensors)
        idx_src_gpu = ti.field(dtype=int, shape=self._pos_sources.shape)
        idx_src_gpu.from_numpy(self._pos_sources)
        idx_sen_gpu = ti.field(dtype=int, shape=self._pos_sensors.shape)
        idx_sen_gpu.from_numpy(self._pos_sensors)
        
        # ---------------------------------
        # Definicao das funcoes de kernel
        # ---------------------------------
        # Pressao
        @ti.kernel
        def pressure_first_der_kernel(one_dx: float, one_dy: float, nx: int, ny: int, ord: int):
            for x, y in pressure_present_gpu:
                last = ord - 1
                offset = ord - 1
                i_dix = -idx_fd_gpu[last, 2]
                i_dfx = nx - idx_fd_gpu[last, 0]
                i_diy = -idx_fd_gpu[last, 2]
                i_dfy = ny - idx_fd_gpu[last, 0]

                # Calculo das primeiras derivadas (forward) da pressao em relacao a x e y
                pressure_l2_norm_gpu[None] = 0.0
                if(x >= i_dix and x < i_dfx and y >= i_diy and y < i_dfy):
                    vdpx_dx = 0.0
                    vdpy_dy = 0.0

                    for c in range(ord * 2):
                        off = c - (ord - 1)
                        vdpx_dx += coefs_gpu[c] * pressure_present_gpu[x + off, y] * one_dx
                        vdpy_dy += coefs_gpu[c] * pressure_present_gpu[x, y + off] * one_dy

                    mdpx_dx_new = b_x_half_gpu[x - offset] * memory_dpressure_dx_gpu[x, y] + a_x_half_gpu[x - offset] * vdpx_dx
                    mdpy_dy_new = b_y_half_gpu[y - offset] * memory_dpressure_dy_gpu[x, y] + a_y_half_gpu[y - offset] * vdpy_dy

                    vdpx_dx = vdpx_dx/k_x_half_gpu[x - offset] + mdpx_dx_new
                    vdpy_dy = vdpy_dy/k_y_half_gpu[y - offset] + mdpy_dy_new

                    memory_dpressure_dx_gpu[x, y] = mdpx_dx_new
                    memory_dpressure_dy_gpu[x, y] = mdpy_dy_new
                    
                    dpressure_dx_gpu[x, y] = vdpx_dx / rho_grid_vx_gpu[x, y]
                    dpressure_dy_gpu[x, y] = vdpy_dy / rho_grid_vy_gpu[x, y]
        
        @ti.kernel
        def pressure_second_der_kernel(dt: float, one_dx: float, one_dy: float, nx: int, ny:int, ord: int, it: int):
            for x, y in pressure_present_gpu:
                last = ord - 1
                offset = ord - 1
                i_dix = -idx_fd_gpu[last, 3]
                i_dfx = nx - idx_fd_gpu[last, 1]
                i_diy = -idx_fd_gpu[last, 3]
                i_dfy = ny - idx_fd_gpu[last, 1]
                p_2_old = pressure_l2_norm_gpu[None]

                # Calculo das segundas derivadas (backward) da pressao em relacao a x e y
                if(x >= i_dix and x < i_dfx and y >= i_diy and y < i_dfy):
                    vdpxx_dx = 0.0
                    vdpyy_dy = 0.0

                    for c in range(ord * 2):
                        ic = (ord * 2 - 1) - c
                        off = (ord - 1) - ic
                        vdpxx_dx += -coefs_gpu[ic] * dpressure_dx_gpu[x + off, y] * one_dx
                        vdpyy_dy += -coefs_gpu[ic] * dpressure_dy_gpu[x, y + off] * one_dy

                    mdpxx_dx_new = b_x_gpu[x - offset] * memory_dpressurexx_dx_gpu[x, y] + a_x_gpu[x - offset] * vdpxx_dx
                    mdpyy_dy_new = b_y_gpu[y - offset] * memory_dpressureyy_dy_gpu[x, y] + a_y_gpu[y - offset] * vdpyy_dy

                    vdpxx_dx = vdpxx_dx/k_x_gpu[x - offset] + mdpxx_dx_new
                    vdpyy_dy = vdpyy_dy/k_y_gpu[y - offset] + mdpyy_dy_new

                    memory_dpressurexx_dx_gpu[x, y] = mdpxx_dx_new
                    memory_dpressureyy_dy_gpu[x, y] = mdpyy_dy_new
                
                    # Atualiza o campo de pressao futuro a partir do passado e do presente
                    pressure_new = 2.0 * pressure_present_gpu[x, y] - pressure_past_gpu[x, y] + dt**2 * (vdpxx_dx + vdpyy_dy) * kappa_gpu[x, y]
                    
                    # Adicao da fonte no campo de pressao
                    idx_src = idx_src_gpu[x, y]
                    if idx_src != -1:
                        pressure_new += source_term_gpu[it - 1, idx_src] * dt**2 * one_dx * one_dy
                        
                    pressure_future_gpu[x, y] = pressure_new
                else:
                    # Condicao de contorno Dirichlet (p = 0) nas bordas
                    pressure_future_gpu[x, y] = 0.0
                
                # Calcula a norma L2 da pressao
                p_2_new = ti.abs(pressure_future_gpu[x, y])
                pressure_l2_norm_gpu[None] = p_2_old if p_2_old > p_2_new else p_2_new
                
                # Swap dos valores novos de pressao para valores antigos
                pressure_past_gpu[x, y] = pressure_present_gpu[x, y]
                pressure_present_gpu[x, y] = pressure_future_gpu[x, y]
            
                # Armazena o sinal do sensor, se o pixel fizer parte de um receptor
                sensor = idx_sen_gpu[x, y]
                if sensor != -1 and it >= delay_rec_gpu[sensor]:
                    sens_pressure_gpu[it - 1, sensor] += pressure_present_gpu[x, y]
              
        # Laco de tempo para execucao da simulacao
        t_gpu = time()
        for it in range(1, self._n_steps + 1):
            # Calculo da pressao
            pressure_first_der_kernel(self._one_dx, self._one_dy, self._nx, self._ny, ord)
                  
            # Calculo das velocidades
            pressure_second_der_kernel(self._dt, self._one_dx, self._one_dy, self._nx, self._ny, ord, it)

            psn2 = pressure_l2_norm_gpu[None]
            if (it % self._it_display) == 0 or it == 5:
                if self._show_debug:
                    print(f'Time step # {it} out of {self._n_steps}')
                    print(f'Max pressure = {psn2}')

                if self._show_anim:
                    self._windows_gpu[0].imv.setImage(pressure_present_gpu.to_numpy()[ix_min:ix_max, iy_min:iy_max],
                                                      levels=[self._min_val_fields, self._max_val_fields])
                    self._app.processEvents()

            # Verifica a estabilidade da simulacao
            if psn2 > STABILITY_THRESHOLD:
                raise StabilityError("Simulacao tornando-se instavel", psn2)
                
        sim_time = time() - t_gpu

        # Pega os resultados da simulacao
        pressure = pressure_present_gpu.to_numpy()
        sens_pressure = sens_pressure_gpu.to_numpy()
        
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
                "gpu_str": ti.lang.impl.current_cfg().arch.name, "sim_time": sim_time}


# ----------------------------------------------------------
# Avaliacao dos parametros na linha de comando
# ----------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', help='Configuration file', default='config.json')
args = parser.parse_args()

# Cria a instancia do simulador
sim_instance = SimulatorTaichiUnsplit(args.config)

# Executa simulacao
try:
    sim_instance.run()

except KeyError as key:
    print(f"Chave {key} nao encontrada no arquivo de configuracao.")
    
except ValueError as value:
    print(value)
