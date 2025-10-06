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
class SimulatorTaichi(Simulator):
    def __init__(self, file_config):
        # Chama do construtor padrao, que le o arquivo de configuracao
        super().__init__(file_config, sim_type = "elastic")

        # Define o nome do simulador
        self._name = "Taichi-elastic"

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
        cp_grid_vx_gpu = ti.field(dtype=float, shape=self._cp_grid_vx.shape)
        cs_grid_vx_gpu = ti.field(dtype=float, shape=self._cs_grid_vx.shape)
        coefs_gpu = ti.field(dtype=float, shape=self._coefs.shape)
        coefs_gpu.from_numpy(self._coefs)

        # Arrays para as variaveis de memoria do calculo
        memory_dvx_dx_gpu = ti.field(dtype=float, shape=(self._nx, self._ny))
        memory_dvx_dy_gpu = ti.field(dtype=float, shape=(self._nx, self._ny))
        memory_dvy_dx_gpu = ti.field(dtype=float, shape=(self._nx, self._ny))
        memory_dvy_dy_gpu = ti.field(dtype=float, shape=(self._nx, self._ny))
        memory_dsigmaxx_dx_gpu = ti.field(dtype=float, shape=(self._nx, self._ny))
        memory_dsigmayy_dy_gpu = ti.field(dtype=float, shape=(self._nx, self._ny))
        memory_dsigmaxy_dx_gpu = ti.field(dtype=float, shape=(self._nx, self._ny))
        memory_dsigmaxy_dy_gpu = ti.field(dtype=float, shape=(self._nx, self._ny))

        # Arrays dos campos de velocidade e pressoes
        vx_gpu = ti.field(dtype=float, shape=(self._nx, self._ny))
        vy_gpu = ti.field(dtype=float, shape=(self._nx, self._ny))
        sigmaxx_gpu = ti.field(dtype=float, shape=(self._nx, self._ny))
        sigmayy_gpu = ti.field(dtype=float, shape=(self._nx, self._ny))
        sigmaxy_gpu = ti.field(dtype=float, shape=(self._nx, self._ny))
        v2_norm_gpu = ti.field(dtype=float, shape=())

        # Arrays para os sensores
        sens_sigxx_gpu = ti.field(dtype=float, shape=(self._n_steps, self._n_rec))
        sens_sigxy_gpu = ti.field(dtype=float, shape=(self._n_steps, self._n_rec))
        sens_sigyy_gpu = ti.field(dtype=float, shape=(self._n_steps, self._n_rec))
        sens_vx_gpu = ti.field(dtype=float, shape=(self._n_steps, self._n_rec))
        sens_vy_gpu = ti.field(dtype=float, shape=(self._n_steps, self._n_rec))
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

        source_term = self._source_term
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
        # Estresse
        @ti.kernel
        def sigma_kernel(dt: float, one_dx: float, one_dy: float, nx: int, ny: int, ord: int, it: int):
            for x, y in sigmayy_gpu:
                last = ord - 1
                offset = ord - 1
                i_dix = -idx_fd_gpu[last, 2]
                i_dfx = nx - idx_fd_gpu[last, 0]
                i_diy = -idx_fd_gpu[last, 3]
                i_dfy = ny - idx_fd_gpu[last, 1]

                #Stress
                v2_norm_gpu[None] = 0.0
                if (x >= i_dix and x < i_dfx and y >= i_diy and y < i_dfy):
                    vdvx_dx = 0.0
                    vdvy_dy = 0.0
                    for c in range(0, ord):
                        vdvx_dx += coefs_gpu[c] * (vx_gpu[x + idx_fd_gpu[c, 0], y] -
                                                   vx_gpu[x + idx_fd_gpu[c, 2], y]) * one_dx
                        vdvy_dy += coefs_gpu[c] * (vy_gpu[x, y + idx_fd_gpu[c, 1]] -
                                                   vy_gpu[x, y + idx_fd_gpu[c, 3]]) * one_dy

                    mdvx_dx_new = b_x_half_gpu[x - offset] * memory_dvx_dx_gpu[x, y] + a_x_half_gpu[
                        x - offset] * vdvx_dx
                    mdvy_dy_new = b_y_gpu[y - offset] * memory_dvy_dy_gpu[x, y] + a_y_gpu[y - offset] * vdvy_dy

                    vdvx_dx = vdvx_dx / k_x_half_gpu[x - offset] + mdvx_dx_new
                    vdvy_dy = vdvy_dy / k_y_gpu[y - offset] + mdvy_dy_new

                    memory_dvx_dx_gpu[x, y] = mdvx_dx_new
                    memory_dvy_dy_gpu[x, y] = mdvy_dy_new

                    rho_h_x = 0.5 * rho_grid_vx_gpu[x+1,y] + rho_grid_vx_gpu[x,y]
                    cp_h_x = 0.5 * cp_grid_vx_gpu[x+1,y] + cp_grid_vx_gpu[x,y]
                    cs_h_x_l = 0.5 * cs_grid_vx_gpu[x+1,y] + cs_grid_vx_gpu[x,y]
                    cs_h_x_m = 0.0 if min(cs_grid_vx_gpu[x+1,y], cs_grid_vx_gpu[x,y]) == 0.0 else cs_h_x_l
                    lambda_gpu = rho_h_x * (cp_h_x * cp_h_x - 2.0 * cs_h_x_l * cs_h_x_l)
                    mu = rho_h_x * (cs_h_x_m * cs_h_x_m)
                    lambdaplus2mu = lambda_gpu + 2.0 * mu
                    lambdaplusmu = lambda_gpu + mu

                    sigmaxx_gpu[x, y] += (lambdaplus2mu * vdvx_dx + lambda_gpu * vdvy_dy) * dt
                    sigmayy_gpu[x,y] += (lambda_gpu * vdvx_dx + lambdaplusmu * vdvy_dy) * dt

                i_dix = -idx_fd_gpu[last, 3]
                i_dfx = nx - idx_fd_gpu[last, 1]
                i_diy = -idx_fd_gpu[last, 2]
                i_dfy = ny - idx_fd_gpu[last, 0]
                if (x >= i_dix and x < i_dfx and y >= i_diy and y < i_dfy):
                    vdvy_dx = 0.0
                    vdvx_dy = 0.0
                    for c in range(0, ord):
                        vdvy_dx += coefs_gpu[c] * (vx_gpu[x + idx_fd_gpu[c, 1], y] -
                                                   vx_gpu[x + idx_fd_gpu[c, 3], y]) * one_dx
                        vdvx_dy += coefs_gpu[c] * (vy_gpu[x, y + idx_fd_gpu[c, 0]] -
                                                   vy_gpu[x, y + idx_fd_gpu[c, 2]]) * one_dy
                    mdvy_dx_new = b_x_gpu[x - offset] * memory_dvy_dx_gpu[x, y] + a_x_gpu[
                        x - offset] * vdvy_dx
                    mdvx_dy_new = b_y_half_gpu[y - offset] * memory_dvx_dy_gpu[x, y] + a_y_half_gpu[y - offset] * vdvx_dy

                    vdvy_dx = vdvy_dx / k_x_gpu[x - offset] + mdvy_dx_new
                    vdvx_dy = vdvx_dy / k_y_half_gpu[y - offset] + mdvx_dy_new

                    memory_dvy_dx_gpu[x, y] = mdvy_dx_new
                    memory_dvx_dy_gpu[x, y] = mdvx_dy_new

                    rho_h_y = 0.5 * (rho_grid_vy_gpu[x, y+1] + rho_grid_vy_gpu[x, y])
                    cs_h_y = 0.0 if min(cs_grid_vx_gpu[x,y+1], cs_grid_vx_gpu[x,y]) == 0.0 else 0.5 * (cs_grid_vx_gpu[x,y+1] + cs_grid_vx_gpu[x,y])
                    mu = rho_h_y * (cs_h_y * cs_h_y)

                    sigmaxy_gpu[x,y] += (vdvx_dy + vdvy_dx) * mu * dt

        # Velocidades
        @ti.kernel
        def velocity_kernel(dt: float, one_dx: float, one_dy: float, nx: int, ny: int, ord: int, it: int):
            for x, y in sigmayy_gpu:
                last = ord - 1
                offset = ord - 1
                i_dix = -idx_fd_gpu[last, 3]
                i_dfx = nx - idx_fd_gpu[last, 1]
                i_diy = -idx_fd_gpu[last, 3]
                i_dfy = ny - idx_fd_gpu[last, 1]
                v_2_old = v2_norm_gpu[None]

                # Velocidade Vx
                if (x >= i_dix and x < i_dfx and y >= i_diy and y < i_dfy):
                    vdsigmaxx_dx = 0.0
                    vdsigmaxy_dy = 0.0
                    for c in range(0, ord):
                        vdsigmaxx_dx += coefs_gpu[c] * (sigmaxx_gpu[x + idx_fd_gpu[c, 1], y] -
                                                        sigmaxx_gpu[x + idx_fd_gpu[c, 3], y]) * one_dx

                        vdsigmaxy_dy += coefs_gpu[c] * (sigmaxy_gpu[x, y + idx_fd_gpu[c, 1]] -
                                                        sigmaxy_gpu[x, y + idx_fd_gpu[c, 3]]) * one_dy

                    mdsxx_dx_new = b_x_gpu[x - offset] * memory_dsigmaxx_dx_gpu[x, y] + a_x_gpu[x - offset] * vdsigmaxx_dx
                    mdsxy_dy_new = b_y_gpu[y - offset] * memory_dsigmaxy_dy_gpu[x, y] + a_y_gpu[y - offset] * vdsigmaxy_dy

                    vdsigmaxx_dx = vdsigmaxx_dx / k_x_gpu[x - offset] + mdsxx_dx_new
                    vdsigmaxy_dy = vdsigmaxy_dy / k_x_gpu[y - offset] + mdsxy_dy_new

                    memory_dsigmaxx_dx_gpu[x, y] = mdsxx_dx_new
                    memory_dsigmaxy_dy_gpu[x, y] = mdsxy_dy_new

                    if rho_grid_vx_gpu[x,y] > 0.0:
                        vx_gpu[x, y] += (vdsigmaxx_dx + vdsigmaxy_dy) * dt / rho_grid_vx_gpu[x,y]

                else:
                    # Condicao de Dirichlet
                    vx_gpu[x, y] = 0.0

                # Velocidade Vy
                i_dix = -idx_fd_gpu[last, 2]
                i_dfx = nx - idx_fd_gpu[last, 0]
                i_diy = -idx_fd_gpu[last, 2]
                i_dfy = ny - idx_fd_gpu[last, 0]

                if (x >= i_dix and x < i_dfx and y >= i_diy and y < i_dfy):
                    vdsigmaxy_dx = 0.0
                    vdsigmayy_dy = 0.0
                    for c in range(0, ord):
                        vdsigmaxy_dx += coefs_gpu[c] * (sigmaxy_gpu[x + idx_fd_gpu[c, 0], y] -
                                                        sigmaxy_gpu[x + idx_fd_gpu[c, 2], y]) * one_dx
                        vdsigmayy_dy += coefs_gpu[c] * (sigmayy_gpu[x, y + idx_fd_gpu[c, 0]] -
                                                        sigmayy_gpu[x, y + idx_fd_gpu[c, 2]]) * one_dy

                    mdsxy_dx_new = b_x_half_gpu[x - offset] * memory_dsigmaxy_dx_gpu[x, y] + a_x_half_gpu[x - offset] * vdsigmaxy_dx
                    mdsyy_dy_new = b_y_half_gpu[y - offset] * memory_dsigmayy_dy_gpu[x, y] + a_y_half_gpu[y - offset] * vdsigmayy_dy

                    vdsigmaxy_dx = vdsigmaxy_dx / k_y_half_gpu[x - offset] + mdsxy_dx_new
                    vdsigmayy_dy = vdsigmayy_dy / k_y_half_gpu[y - offset] + mdsyy_dy_new

                    memory_dsigmaxy_dx_gpu[x, y] = mdsxy_dx_new
                    memory_dsigmayy_dy_gpu[x,y] = mdsyy_dy_new

                    rho = 0.25 * rho_grid_vx_gpu[x,y] + rho_grid_vx_gpu[x + 1, y] + rho_grid_vy_gpu[x + 1,y + 1] + rho_grid_vy_gpu[x, y + 1]
                    if rho > 0.0:
                        vy_gpu[x, y] += (vdsigmaxy_dx + vdsigmayy_dy) * dt / rho
                else:
                    # Condicao de Dirichlet
                    vy_gpu[x, y] = 0.0

                # Adiciona o sinal de fonte, se o pixel fizer parte de uma fonte
                idx_src = idx_src_gpu[x, y]
                rho = 0.25 * rho_grid_vx_gpu[x, y] + rho_grid_vx_gpu[x + 1, y] + rho_grid_vy_gpu[x + 1, y + 1] + rho_grid_vy_gpu[x, y + 1]
                if (idx_src != -1 and rho > 0.0):
                    vy_gpu[x, y] += source_term_gpu[it - 1, idx_src] * dt / rho

                # Calcula a velocidade normal L2
                v_2_new = ti.abs(vx_gpu[x,y] * vx_gpu[x,y] + vy_gpu[x,y] * vy_gpu[x,y])
                v2_norm_gpu[None] = v_2_old if v_2_old > v_2_new else v_2_new

                # Armazena o sinal do sensor, se o pixel fizer parte de um receptor
                sensor = idx_sen_gpu[x, y]
                if sensor != -1 and it >= delay_rec_gpu[sensor]:
                    sens_vx_gpu[it - 1, sensor] += vx_gpu[x, y]
                    sens_vy_gpu[it - 1, sensor] += vy_gpu[x, y]

                    sens_sigxx_gpu[it - 1, sensor] += sigmaxx_gpu[x, y]
                    sens_sigxy_gpu[it - 1, sensor] += sigmaxy_gpu[x, y]
                    sens_sigyy_gpu[it - 1, sensor] += sigmayy_gpu[x, y]




        # Laco de tempo para execucao da simulacao
        t_gpu = time()
        for it in range(1, self._n_steps + 1):
            # Calculo dos estresses
            sigma_kernel(self._dt, self._one_dx, self._one_dy, self._nx, self._ny, ord, it)

            # Calculo das velocidades
            velocity_kernel(self._dt, self._one_dx, self._one_dy, self._nx, self._ny, ord, it)

            vsn2 = np.sqrt(v2_norm_gpu[None])
            if (it % self._it_display) == 0 or it == 5:
                if self._show_debug:
                    print(f'Time step # {it} out of {self._n_steps}')
                    print(f'Max norm velocity vector V (m/s) = {vsn2}')

                if self._show_anim:
                    self._windows_gpu[0].imv.setImage(vx_gpu.to_numpy()[ix_min:ix_max, iy_min:iy_max],
                                                      levels=[self._min_val_fields, self._max_val_fields])
                    self._windows_gpu[1].imv.setImage(vy_gpu.to_numpy()[ix_min:ix_max, iy_min:iy_max],
                                                      levels=[self._min_val_fields, self._max_val_fields])
                    self._app.processEvents()

            # Verifica a estabilidade da simulacao
            if vsn2 > STABILITY_THRESHOLD:
                raise StabilityError("Simulacao tornando-se instavel", vsn2)

        sim_time = time() - t_gpu

        # Pega os resultados da simulacao
        sigmayy = sigmayy_gpu.to_numpy()
        sens_sigyy = sens_sigyy_gpu.to_numpy()

        # --------------------------------------------
        # A funcao de implementacao do simulador deve retornar
        # um dicionario com as seguintes chaves:
        #   - "stress": campo de estresse
        #   - "sens_stress": sinais da estresse nos sensores
        #   - "gpu_str": string de identificacao da GPU utilizada na simulacao
        #   - "sim_time": tempo da simulacao, medido com a funcao time()
        #   - opcionalmente pode ter uma mensagem exclusiva da implementacao em "msg_impl"
        # --------------------------------------------
        return {"stress": sigmayy, "sens_stress": sens_sigyy,
                "gpu_str": ti.lang.impl.current_cfg().arch.name, "sim_time": sim_time}


# ----------------------------------------------------------
# Avaliacao dos parametros na linha de comando
# ----------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', help='Configuration file', default='config.json')
args = parser.parse_args()

# Cria a instancia do simulador
sim_instance = SimulatorTaichi(args.config)

# Executa simulacao
try:
    sim_instance.run()

except KeyError as key:
    print(f"Chave {key} nao encontrada no arquivo de configuracao.")

except ValueError as value:
    print(value)
