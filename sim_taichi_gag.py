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
class SimulatorTaichiGAG(Simulator):
    def __init__(self, file_config):
        # Chama do construtor padrao, que le o arquivo de configuracao
        super().__init__(file_config)
        
        # Define o nome do simulador
        self._name = "Taichi-GAG"
        
        
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
        memory_dvx_dx_gpu = ti.field(dtype=float, shape=(self._nx, self._ny))
        memory_dvy_dy_gpu = ti.field(dtype=float, shape=(self._nx, self._ny))
        memory_dpressure_dx_gpu = ti.field(dtype=float, shape=(self._nx, self._ny))
        memory_dpressure_dy_gpu = ti.field(dtype=float, shape = (self._nx, self._ny))

        # Arrays dos campos de velocidade e pressoes
        vx_gpu = ti.field(dtype=float, shape=(self._nx, self._ny))
        vy_gpu = ti.field(dtype=float, shape=(self._nx, self._ny))
        pressure_gpu = ti.field(dtype=float, shape=(self._nx, self._ny))
        pressure_l2_norm_gpu = ti.field(dtype=float, shape=())
        
        # Arrays para os sensores
        sens_pressure_gpu = ti.field(dtype=float, shape=(self._n_steps, self._n_rec))
        delay_rec_gpu = ti.field(dtype=int, shape=self._delay_recv.shape)
        delay_rec_gpu.from_numpy(self._delay_recv)

        # Calculo dos indices para o staggered grid
        ord = self._coefs.shape[0]
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
        source_term_gpu.from_numpy(source_term * (self._dt * self._one_dx * self._one_dy))
        
        # Arrays com as informacoes sobre os elementos emissores (sources) e receptores (sensors)
        idx_src_gpu = ti.field(dtype=int, shape=self._pos_sources.shape)
        idx_src_gpu.from_numpy(self._pos_sources)
        idx_sen_gpu = ti.field(dtype=int, shape=self._pos_sensors.shape)
        idx_sen_gpu.from_numpy(self._pos_sensors)
        
        # ---------------------------------
        # Definicao das funcoes de kernel
        # ---------------------------------
        @ti.func
        def D(u: ti.template(), xyz, nd: int, bf: int, imax: int): # type: ignore
            """
            Derivative operator

            Parameters
            ----------
            u: field
            xyz: coordinate
            nd: dimension (0: x, 1: y, 2: z)
            bf: backward (0) or forward (1)
            imax: maximum index

            Returns
            -------
            d: derivative of field
            """
            d = 0.
            for nc in ti.static(range(self._deriv_acc)):
                xyz[nd] += nc + bf
                a = u[xyz] if xyz[nd] < imax else 0
                xyz[nd] += - 2 * nc - 1
                b = u[xyz] if xyz[nd] >= 0 else 0
                xyz[nd] += nc + 1 - bf
                d += self._coefs[nc] * (a - b)

            return d
        
        # Pressao
        @ti.kernel
        def pressure_kernel(dt: float, one_dx: float, one_dy: float, nx: int, ny: int, ord: int, it: int):
            for xy in ti.grouped(pressure_gpu):
                x = xy[0]
                y = xy[1]
                last = ord - 1
                offset = ord - 1
                i_dix = -idx_fd_gpu[last, 2]
                i_dfx = nx - idx_fd_gpu[last, 0]
                i_diy = -idx_fd_gpu[last, 3]
                i_dfy = ny - idx_fd_gpu[last, 1]

                # Pressure
                pressure_l2_norm_gpu[None] = 0.0
                if(x >= i_dix and x < i_dfx and y >= i_diy and y < i_dfy):
                    vdvx_dx = D(vx_gpu, xy, 0, 0, nx) * one_dx
                    vdvy_dy = D(vy_gpu, xy, 1, 0, ny) * one_dy

                    mdvx_dx_new = b_x_half_gpu[x - offset] * memory_dvx_dx_gpu[xy] + a_x_half_gpu[x - offset] * vdvx_dx
                    mdvy_dy_new = b_y_gpu[y - offset] * memory_dvy_dy_gpu[xy] + a_y_gpu[y - offset] * vdvy_dy

                    vdvx_dx = vdvx_dx/k_x_half_gpu[x - offset] + mdvx_dx_new
                    vdvy_dy = vdvy_dy/k_y_gpu[y - offset]  + mdvy_dy_new

                    memory_dvx_dx_gpu[xy] = mdvx_dx_new
                    memory_dvy_dy_gpu[xy] = mdvy_dy_new

                    pressure_gpu[xy] += kappa_gpu[xy]*(vdvx_dx + vdvy_dy) * dt
                    
                # Adiciona o sinal de fonte, se o pixel fizer parte de uma fonte
                idx_src = idx_src_gpu[xy]
                if idx_src != -1:
                    pressure_gpu[xy] += source_term_gpu[it - 1, idx_src]
        
        # Velocidades
        @ti.kernel
        def velocity_kernel(dt: float, one_dx: float, one_dy: float, nx: int, ny: int, ord: int, it: int):
            for xy in ti.grouped(pressure_gpu):
                x = xy[0]
                y = xy[1]
                last = ord - 1
                offset = ord - 1
                i_dix = -idx_fd_gpu[last, 3]
                i_dfx = nx - idx_fd_gpu[last, 1]
                i_diy = -idx_fd_gpu[last, 3]
                i_dfy = ny - idx_fd_gpu[last, 1]
                p_2_old = pressure_l2_norm_gpu[None]
                
                # Velocidade Vx
                if(x >= i_dix and x < i_dfx and y >= i_diy and y < i_dfy):
                    dpressure_dx = D(pressure_gpu, xy, 0, 1, nx) * one_dx

                    mdpressure_dx_new = b_x_gpu[x - offset] * memory_dpressure_dx_gpu[xy] + a_x_gpu[x - offset] * dpressure_dx
                    dpressure_dx = dpressure_dx / k_x_gpu[x - offset] + mdpressure_dx_new
                    memory_dpressure_dx_gpu[xy] = mdpressure_dx_new

                    vx_gpu[xy] += dt * (dpressure_dx / rho_grid_vx_gpu[x, y])
                else:
                    # Condicao de Dirichlet
                    vx_gpu[xy] = 0.0
                
                # Velocidade Vy
                i_dix = -idx_fd_gpu[last, 2]
                i_dfx = nx - idx_fd_gpu[last, 0]
                i_diy = -idx_fd_gpu[last, 2]
                i_dfy = ny - idx_fd_gpu[last, 0]
                
                if(x >= i_dix and x < i_dfx and y >= i_diy and y < i_dfy):
                    dpressure_dy = D(pressure_gpu, xy, 1, 1, ny) * one_dy

                    mdpressure_dy_new = b_y_half_gpu[y - offset] * memory_dpressure_dy_gpu[xy] + a_y_half_gpu[y - offset] * dpressure_dy
                    dpressure_dy = dpressure_dy / k_y_half_gpu[y - offset] + mdpressure_dy_new
                    memory_dpressure_dy_gpu[xy] = mdpressure_dy_new

                    vy_gpu[xy] += dt * (dpressure_dy / rho_grid_vy_gpu[xy])
                else:
                    # Condicao de Dirichlet
                    vy_gpu[xy] = 0.0
                    
                # Calcula a norma L2 da pressao
                p_2_new = ti.abs(pressure_gpu[xy])
                pressure_l2_norm_gpu[None] = p_2_old if p_2_old > p_2_new else p_2_new
                
                # Armazena o sinal do sensor, se o pixel fizer parte de um receptor
                sensor = idx_sen_gpu[xy]
                if sensor != -1 and it >= delay_rec_gpu[sensor]:
                    sens_pressure_gpu[it - 1, sensor] += pressure_gpu[xy]
              
        # Laco de tempo para execucao da simulacao
        t_gpu = time()
        for it in range(1, self._n_steps + 1):
            # Calculo da pressao
            pressure_kernel(self._dt, self._one_dx, self._one_dy, self._nx, self._ny, ord, it)
                  
            # Calculo das velocidades
            velocity_kernel(self._dt, self._one_dx, self._one_dy, self._nx, self._ny, ord, it)
            
            psn2 = pressure_l2_norm_gpu[None]
            if (it % self._it_display) == 0 or it == 5:
                if self._show_debug:
                    print(f'Time step # {it} out of {self._n_steps}')
                    print(f'Max pressure = {psn2}')

                if self._show_anim:
                    self._windows_gpu[0].imv.setImage(pressure_gpu.to_numpy()[ix_min:ix_max, iy_min:iy_max],
                                                      levels=[self._min_val_fields, self._max_val_fields])
                    self._app.processEvents()

            # Verifica a estabilidade da simulacao
            if psn2 > STABILITY_THRESHOLD:
                raise StabilityError("Simulacao tornando-se instavel", psn2)
                
        sim_time = time() - t_gpu
        #ti.profiler.print_kernel_profiler_info()

        # Pega os resultados da simulacao
        pressure = pressure_gpu.to_numpy()
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
sim_instance = SimulatorTaichiGAG(args.config)

# Executa simulacao
try:
    sim_instance.run()

except KeyError as key:
    print(f"Chave {key} nao encontrada no arquivo de configuracao.")
    
except ValueError as value:
    print(value)
