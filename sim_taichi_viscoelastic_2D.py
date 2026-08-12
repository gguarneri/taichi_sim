# =======================
# Importacao de pacotes de uso geral
# =======================
import argparse
from time import time
import numpy as np
from sim_support import *
from sim_support.attenuation import AttenuationCoefficients
from sim_support.simulator import Simulator

# ======================
# Importacao do framework Taichi
# ======================
import taichi as ti


# -----------------------------------------------------------------------------
# Simulador de ondas viscoelasticas 2D usando o framework Taichi
# -----------------------------------------------------------------------------
class SimulatorTaichiViscoelastic(Simulator):
    def __init__(self, file_config):
        # Chama o construtor padrao, que le o arquivo de configuracao
        super().__init__(file_config, sim_type="viscoelastic")

        # Define o nome do simulador
        self._name = "Taichi-viscoelastic"

    def implementation(self):
        super().implementation()

        # --------------------------------------------
        # Inicializacao do framework Taichi
        # --------------------------------------------
        ti.init(arch=ti.gpu, debug=False, print_ir=False, kernel_profiler=False)

        # --------------------------------------------
        # Parametros gerais da simulacao
        # --------------------------------------------
        nx  = self._nx
        ny  = self._ny
        dt  = float(self._dt)
        dx  = float(self._roi.get_dx())
        dy  = float(self._roi.get_dz())
        one_dx = 1.0 / dx
        one_dy = 1.0 / dy
        ord = self._coefs.shape[0]

        # Coeficientes de atenuacao viscoelastica (modelo SLS)
        att      = AttenuationCoefficients()
        n_sls    = att._n_sls
        visco_on = bool(self._configs["simul_configs"]["viscoelastic_attn"])

        # sum_alpha: [sum_alpha_p, sum_alpha_s]
        alpha_sum_np = np.array([att._sum_alpha_p, att._sum_alpha_s], dtype=np.float32)

        # tau_att: linhas = [tau_eps_p, tau_sig_p, tau_eps_s, tau_sig_s], colunas = corpo SLS
        tau_att_np = np.array(
            [att._tau_epsilon_p,
             att._tau_sigma_p,
             att._tau_epsilon_s,
             att._tau_sigma_s],
            dtype=np.float32
        )  # shape (4, n_sls)

        # Indices do staggered grid: [c,0]=ih, [c,1]=if_, [c,2]=fh, [c,3]=ff
        idx_fd_np = np.array([[c + 1, c, -c, -c - 1] for c in range(ord)], dtype=np.int32)

        # Acrescenta eixo se source_term for array unidimensional
        if self._n_pto_src == 1:
            source_term = self._source_term[:, np.newaxis]
        else:
            source_term = self._source_term

        # --------------------------------------------
        # Declaracao dos fields Taichi
        # --------------------------------------------

        # --- Coeficientes CPML (tamanho real dos arrays da classe base) ---
        a_x_gpu      = ti.field(dtype=ti.f32, shape=self._a_x.flatten().shape)
        b_x_gpu      = ti.field(dtype=ti.f32, shape=self._b_x.flatten().shape)
        k_x_gpu      = ti.field(dtype=ti.f32, shape=self._k_x.flatten().shape)
        a_x_half_gpu = ti.field(dtype=ti.f32, shape=self._a_x_half.flatten().shape)
        b_x_half_gpu = ti.field(dtype=ti.f32, shape=self._b_x_half.flatten().shape)
        k_x_half_gpu = ti.field(dtype=ti.f32, shape=self._k_x_half.flatten().shape)

        a_y_gpu      = ti.field(dtype=ti.f32, shape=self._a_y.flatten().shape)
        b_y_gpu      = ti.field(dtype=ti.f32, shape=self._b_y.flatten().shape)
        k_y_gpu      = ti.field(dtype=ti.f32, shape=self._k_y.flatten().shape)
        a_y_half_gpu = ti.field(dtype=ti.f32, shape=self._a_y_half.flatten().shape)
        b_y_half_gpu = ti.field(dtype=ti.f32, shape=self._b_y_half.flatten().shape)
        k_y_half_gpu = ti.field(dtype=ti.f32, shape=self._k_y_half.flatten().shape)

        # --- Mapas do meio fisico ---
        rho_gpu = ti.field(dtype=ti.f32, shape=(nx, ny))
        cp_gpu  = ti.field(dtype=ti.f32, shape=(nx, ny))
        cs_gpu  = ti.field(dtype=ti.f32, shape=(nx, ny))

        # --- Coeficientes de diferenca finita ---
        coefs_gpu  = ti.field(dtype=ti.f32, shape=(ord,))
        idx_fd_gpu = ti.field(dtype=ti.i32, shape=(ord, 4))

        # --- Atenuacao viscoelastica ---
        alpha_sum_gpu = ti.field(dtype=ti.f32, shape=(2,))
        tau_att_gpu   = ti.field(dtype=ti.f32, shape=(4, n_sls))

        # --- Campos de velocidade ---
        vx_gpu        = ti.field(dtype=ti.f32, shape=(nx, ny))
        vy_gpu        = ti.field(dtype=ti.f32, shape=(nx, ny))
        v_l2_norm_gpu = ti.field(dtype=ti.f32, shape=())

        # --- Campos de tensao ---
        sigmaxx_gpu = ti.field(dtype=ti.f32, shape=(nx, ny))
        sigmayy_gpu = ti.field(dtype=ti.f32, shape=(nx, ny))
        sigmaxy_gpu = ti.field(dtype=ti.f32, shape=(nx, ny))

        # --- Variaveis de memoria CPML ---
        memory_dvx_dx_gpu      = ti.field(dtype=ti.f32, shape=(nx, ny))
        memory_dvx_dy_gpu      = ti.field(dtype=ti.f32, shape=(nx, ny))
        memory_dvy_dx_gpu      = ti.field(dtype=ti.f32, shape=(nx, ny))
        memory_dvy_dy_gpu      = ti.field(dtype=ti.f32, shape=(nx, ny))
        memory_dsigmaxx_dx_gpu = ti.field(dtype=ti.f32, shape=(nx, ny))
        memory_dsigmayy_dy_gpu = ti.field(dtype=ti.f32, shape=(nx, ny))
        memory_dsigmaxy_dx_gpu = ti.field(dtype=ti.f32, shape=(nx, ny))
        memory_dsigmaxy_dy_gpu = ti.field(dtype=ti.f32, shape=(nx, ny))

        # --- Variaveis de relaxacao viscoelastica (SLS) ---
        r_xx_gpu     = ti.field(dtype=ti.f32, shape=(nx, ny, n_sls))
        r_yy_gpu     = ti.field(dtype=ti.f32, shape=(nx, ny, n_sls))
        r_xy_gpu     = ti.field(dtype=ti.f32, shape=(nx, ny, n_sls))
        r_xx_old_gpu = ti.field(dtype=ti.f32, shape=(nx, ny, n_sls))
        r_yy_old_gpu = ti.field(dtype=ti.f32, shape=(nx, ny, n_sls))
        r_xy_old_gpu = ti.field(dtype=ti.f32, shape=(nx, ny, n_sls))

        # --- Fonte e receptores ---
        source_term_gpu = ti.field(dtype=ti.f32, shape=source_term.shape)
        idx_src_gpu     = ti.field(dtype=ti.i32, shape=self._pos_sources.shape)
        idx_sen_gpu     = ti.field(dtype=ti.i32, shape=self._pos_sensors.shape)
        delay_rec_gpu   = ti.field(dtype=ti.i32, shape=self._delay_recv.shape)

        # --- Sinais dos sensores ---
        sens_sigyy_gpu = ti.field(dtype=ti.f32, shape=(self._n_steps, self._n_rec))

        # --------------------------------------------
        # Preenchimento dos fields com dados numpy
        # --------------------------------------------
        a_x_gpu.from_numpy(self._a_x.flatten().astype(np.float32))
        b_x_gpu.from_numpy(self._b_x.flatten().astype(np.float32))
        k_x_gpu.from_numpy(self._k_x.flatten().astype(np.float32))
        a_x_half_gpu.from_numpy(self._a_x_half.flatten().astype(np.float32))
        b_x_half_gpu.from_numpy(self._b_x_half.flatten().astype(np.float32))
        k_x_half_gpu.from_numpy(self._k_x_half.flatten().astype(np.float32))

        a_y_gpu.from_numpy(self._a_y.flatten().astype(np.float32))
        b_y_gpu.from_numpy(self._b_y.flatten().astype(np.float32))
        k_y_gpu.from_numpy(self._k_y.flatten().astype(np.float32))
        a_y_half_gpu.from_numpy(self._a_y_half.flatten().astype(np.float32))
        b_y_half_gpu.from_numpy(self._b_y_half.flatten().astype(np.float32))
        k_y_half_gpu.from_numpy(self._k_y_half.flatten().astype(np.float32))

        rho_gpu.from_numpy(self._rho_grid_vx.astype(np.float32))
        cp_gpu.from_numpy(self._cp_grid_vx.astype(np.float32))
        cs_gpu.from_numpy(self._cs_grid_vx.astype(np.float32))

        coefs_gpu.from_numpy(self._coefs.astype(np.float32))
        idx_fd_gpu.from_numpy(idx_fd_np)

        alpha_sum_gpu.from_numpy(alpha_sum_np)
        tau_att_gpu.from_numpy(tau_att_np)

        source_term_gpu.from_numpy(source_term.astype(np.float32))
        idx_src_gpu.from_numpy(self._pos_sources)
        idx_sen_gpu.from_numpy(self._pos_sensors)
        delay_rec_gpu.from_numpy(self._delay_recv)

        # --------------------------------------------
        # Definicao dos limites para plotagem
        # --------------------------------------------
        ix_min = self._roi.get_ix_min()
        ix_max = self._roi.get_ix_max()
        iy_min = self._roi.get_iz_min()
        iy_max = self._roi.get_iz_max()

        # ============================================================
        # KERNEL: sigma_kernel
        # Calcula as tensoes normais (sigmaxx, sigmayy) e
        # cisalhante (sigmaxy), com ou sem atenuacao viscoelastica
        # ============================================================
        @ti.kernel
        def sigma_kernel(dt: float, one_dx: float, one_dy: float,
                         nx: int, ny: int, ord: int, it: int,
                         visco_on: int):
            for x, y in sigmaxx_gpu:
                last   = ord - 1
                offset = ord - 1

                # --------------------------------------------------
                # Tensoes normais: sigmaxx e sigmayy
                # Dominio: staggered half em x, full em y
                # --------------------------------------------------
                i_dix = -idx_fd_gpu[last, 2]   # -get_idx_fh(last)
                i_dfx =  nx - idx_fd_gpu[last, 0]  # nx - get_idx_ih(last)
                i_diy = -idx_fd_gpu[last, 3]   # -get_idx_ff(last)
                i_dfy =  ny - idx_fd_gpu[last, 1]  # ny - get_idx_if(last)

                if x >= i_dix and x < i_dfx and y >= i_diy and y < i_dfy:
                    # Derivadas das velocidades (antes do CPML)
                    vdvx_dx = 0.0
                    vdvy_dy = 0.0
                    for c in range(ord):
                        vdvx_dx += coefs_gpu[c] * (vx_gpu[x + idx_fd_gpu[c, 0], y] -
                                                    vx_gpu[x + idx_fd_gpu[c, 2], y]) * one_dx
                        vdvy_dy += coefs_gpu[c] * (vy_gpu[x, y + idx_fd_gpu[c, 1]] -
                                                    vy_gpu[x, y + idx_fd_gpu[c, 3]]) * one_dy

                    # Atualizacao das variaveis de memoria CPML
                    mdvx_dx_new = b_x_half_gpu[x - offset] * memory_dvx_dx_gpu[x, y] + a_x_half_gpu[x - offset] * vdvx_dx
                    mdvy_dy_new = b_y_gpu[y - offset]      * memory_dvy_dy_gpu[x, y] + a_y_gpu[y - offset]      * vdvy_dy

                    # Derivadas corrigidas pelo CPML
                    vdvx_dx = vdvx_dx / k_x_half_gpu[x - offset] + mdvx_dx_new
                    vdvy_dy = vdvy_dy / k_y_gpu[y - offset]      + mdvy_dy_new

                    memory_dvx_dx_gpu[x, y] = mdvx_dx_new
                    memory_dvy_dy_gpu[x, y] = mdvy_dy_new

                    # Parametros elasticos no ponto (media harmonica em x)
                    rho_hx    = 0.5 * (rho_gpu[x + 1, y] + rho_gpu[x, y])
                    cp_hx     = 0.5 * (cp_gpu[x + 1, y]  + cp_gpu[x, y])
                    cs_hx_l   = 0.5 * (cs_gpu[x + 1, y]  + cs_gpu[x, y])
                    cs_hx_m   = 0.0 if ti.min(cs_gpu[x + 1, y], cs_gpu[x, y]) == 0.0 else cs_hx_l
                    lmbda     = rho_hx * (cp_hx * cp_hx - 2.0 * cs_hx_l * cs_hx_l)
                    mu        = rho_hx * (cs_hx_m * cs_hx_m)
                    lam2mu    = lmbda + 2.0 * mu
                    lammu     = lmbda + mu

                    if visco_on == 1:
                        # ---- Caminho viscoelastico (modelo SLS) ----
                        sum_r_xx = 0.0
                        sum_r_yy = 0.0

                        for _l in range(n_sls):
                            inv_tau_sig_p = 1.0 / tau_att_gpu[1, _l]
                            inv_tau_sig_s = 1.0 / tau_att_gpu[3, _l]
                            alpha_p = tau_att_gpu[0, _l] / tau_att_gpu[1, _l]
                            alpha_s = tau_att_gpu[2, _l] / tau_att_gpu[3, _l]

                            dphi_p = dt * (1.0 - alpha_p) * inv_tau_sig_p / alpha_sum_gpu[0]
                            dphi_s = dt * (1.0 - alpha_s) * inv_tau_sig_s / alpha_sum_gpu[1]

                            hdt_p  = 0.5 * dt * inv_tau_sig_p
                            hdt_s  = 0.5 * dt * inv_tau_sig_s
                            mf_p   = 1.0 / (1.0 + hdt_p)
                            mf_s   = 1.0 / (1.0 + hdt_s)

                            rxx_old = r_xx_gpu[x, y, _l]
                            ryy_old = r_yy_gpu[x, y, _l]

                            rxx = (rxx_old + (vdvx_dx + vdvy_dy) * dphi_p - rxx_old * hdt_p) * mf_p
                            ryy = (ryy_old + 0.5 * (vdvx_dx - vdvy_dy) * dphi_s - ryy_old * hdt_s) * mf_s

                            sum_r_xx += rxx_old + rxx
                            sum_r_yy += ryy_old + ryy

                            r_xx_old_gpu[x, y, _l] = rxx_old
                            r_xx_gpu[x, y, _l]     = rxx
                            r_yy_old_gpu[x, y, _l] = ryy_old
                            r_yy_gpu[x, y, _l]     = ryy

                        sigmaxx_gpu[x, y] += (lam2mu * vdvx_dx + lmbda * vdvy_dy +
                                              0.5 * lammu * sum_r_xx + mu * sum_r_yy) * dt
                        sigmayy_gpu[x, y] += (lmbda * vdvx_dx + lam2mu * vdvy_dy +
                                              0.5 * lammu * sum_r_xx - mu * sum_r_yy) * dt
                    else:
                        # ---- Caminho elastico puro ----
                        sigmaxx_gpu[x, y] += (lam2mu * vdvx_dx + lmbda * vdvy_dy) * dt
                        sigmayy_gpu[x, y] += (lmbda * vdvx_dx + lam2mu * vdvy_dy) * dt

                    # Armazena sinal do sensor (se o pixel pertencer a um receptor)
                    sensor = idx_sen_gpu[x, y]
                    if sensor != -1 and it >= delay_rec_gpu[sensor]:
                        sens_sigyy_gpu[it - 1, sensor] += sigmayy_gpu[x, y]

                # --------------------------------------------------
                # Tensao cisalhante: sigmaxy
                # Dominio: full em x, staggered half em y
                # --------------------------------------------------
                i_dix2 = -idx_fd_gpu[last, 3]   # -get_idx_ff(last)
                i_dfx2 =  nx - idx_fd_gpu[last, 1]  # nx - get_idx_if(last)
                i_diy2 = -idx_fd_gpu[last, 2]   # -get_idx_fh(last)
                i_dfy2 =  ny - idx_fd_gpu[last, 0]  # ny - get_idx_ih(last)

                if x >= i_dix2 and x < i_dfx2 and y >= i_diy2 and y < i_dfy2:
                    vdvy_dx = 0.0
                    vdvx_dy = 0.0
                    for c in range(ord):
                        vdvy_dx += coefs_gpu[c] * (vy_gpu[x + idx_fd_gpu[c, 1], y] -
                                                    vy_gpu[x + idx_fd_gpu[c, 3], y]) * one_dx
                        vdvx_dy += coefs_gpu[c] * (vx_gpu[x, y + idx_fd_gpu[c, 0]] -
                                                    vx_gpu[x, y + idx_fd_gpu[c, 2]]) * one_dy

                    mdvy_dx_new = b_x_gpu[x - offset]      * memory_dvy_dx_gpu[x, y] + a_x_gpu[x - offset]      * vdvy_dx
                    mdvx_dy_new = b_y_half_gpu[y - offset]  * memory_dvx_dy_gpu[x, y] + a_y_half_gpu[y - offset] * vdvx_dy

                    vdvy_dx = vdvy_dx / k_x_gpu[y - offset]      + mdvy_dx_new
                    vdvx_dy = vdvx_dy / k_y_half_gpu[y - offset] + mdvx_dy_new

                    memory_dvy_dx_gpu[x, y] = mdvy_dx_new
                    memory_dvx_dy_gpu[x, y] = mdvx_dy_new

                    rho_hy = 0.5 * (rho_gpu[x, y + 1] + rho_gpu[x, y])
                    cs_hy  = 0.0 if ti.min(cs_gpu[x, y + 1], cs_gpu[x, y]) == 0.0 \
                             else 0.5 * (cs_gpu[x, y + 1] + cs_gpu[x, y])
                    mu_xy  = rho_hy * cs_hy * cs_hy

                    if visco_on == 1:
                        sum_r_xy = 0.0

                        for _l in range(n_sls):
                            inv_tau_sig_s = 1.0 / tau_att_gpu[3, _l]
                            alpha_s = tau_att_gpu[2, _l] / tau_att_gpu[3, _l]
                            dphi_s  = dt * (1.0 - alpha_s) * inv_tau_sig_s / alpha_sum_gpu[1]
                            hdt_s   = 0.5 * dt * inv_tau_sig_s
                            mf_s    = 1.0 / (1.0 + hdt_s)

                            rxy_old = r_xy_gpu[x, y, _l]
                            rxy = (rxy_old + (vdvy_dx + vdvx_dy) * dphi_s - rxy_old * hdt_s) * mf_s

                            sum_r_xy += rxy_old + rxy

                            r_xy_old_gpu[x, y, _l] = rxy_old
                            r_xy_gpu[x, y, _l]     = rxy

                        sigmaxy_gpu[x, y] += (mu_xy * (vdvy_dx + vdvx_dy) +
                                              0.5 * mu_xy * sum_r_xy) * dt
                    else:
                        sigmaxy_gpu[x, y] += (vdvx_dy + vdvy_dx) * mu_xy * dt

        # ============================================================
        # KERNEL: velocity_kernel
        # Calcula as velocidades vx e vy, adiciona a fonte em vy e
        # computa a norma L2 maxima
        # ============================================================
        @ti.kernel
        def velocity_kernel(dt: float, one_dx: float, one_dy: float,
                            nx: int, ny: int, ord: int, it: int):
            for x, y in vx_gpu:
                last   = ord - 1
                offset = ord - 1

                # --------------------------------------------------
                # Velocidade Vx
                # Dominio: full em x e y
                # --------------------------------------------------
                i_dix = -idx_fd_gpu[last, 3]   # -get_idx_ff(last)
                i_dfx =  nx - idx_fd_gpu[last, 1]  # nx - get_idx_if(last)
                i_diy = -idx_fd_gpu[last, 3]
                i_dfy =  ny - idx_fd_gpu[last, 1]

                if x >= i_dix and x < i_dfx and y >= i_diy and y < i_dfy:
                    dsigmaxx_dx = 0.0
                    dsigmaxy_dy = 0.0
                    for c in range(ord):
                        dsigmaxx_dx += coefs_gpu[c] * (sigmaxx_gpu[x + idx_fd_gpu[c, 1], y] -
                                                        sigmaxx_gpu[x + idx_fd_gpu[c, 3], y]) * one_dx
                        dsigmaxy_dy += coefs_gpu[c] * (sigmaxy_gpu[x, y + idx_fd_gpu[c, 1]] -
                                                        sigmaxy_gpu[x, y + idx_fd_gpu[c, 3]]) * one_dy

                    mdsxx_dx_new = b_x_gpu[x - offset] * memory_dsigmaxx_dx_gpu[x, y] + a_x_gpu[x - offset] * dsigmaxx_dx
                    mdsxy_dy_new = b_y_gpu[y - offset] * memory_dsigmaxy_dy_gpu[x, y] + a_y_gpu[y - offset] * dsigmaxy_dy

                    dsigmaxx_dx = dsigmaxx_dx / k_x_gpu[x - offset] + mdsxx_dx_new
                    dsigmaxy_dy = dsigmaxy_dy / k_y_gpu[y - offset] + mdsxy_dy_new

                    memory_dsigmaxx_dx_gpu[x, y] = mdsxx_dx_new
                    memory_dsigmaxy_dy_gpu[x, y] = mdsxy_dy_new

                    rho = rho_gpu[x, y]
                    if rho > 0.0:
                        vx_gpu[x, y] += (dsigmaxx_dx + dsigmaxy_dy) * dt / rho
                else:
                    # Condicao de Dirichlet
                    vx_gpu[x, y] = 0.0

                # --------------------------------------------------
                # Velocidade Vy
                # Dominio: staggered half em x e y
                # --------------------------------------------------
                i_dix2 = -idx_fd_gpu[last, 2]   # -get_idx_fh(last)
                i_dfx2 =  nx - idx_fd_gpu[last, 0]  # nx - get_idx_ih(last)
                i_diy2 = -idx_fd_gpu[last, 2]
                i_dfy2 =  ny - idx_fd_gpu[last, 0]

                rho4 = 0.25 * (rho_gpu[x, y]     + rho_gpu[x + 1, y] +
                               rho_gpu[x + 1, y + 1] + rho_gpu[x, y + 1])

                if x >= i_dix2 and x < i_dfx2 and y >= i_diy2 and y < i_dfy2:
                    dsigmaxy_dx = 0.0
                    dsigmayy_dy = 0.0
                    for c in range(ord):
                        dsigmaxy_dx += coefs_gpu[c] * (sigmaxy_gpu[x + idx_fd_gpu[c, 0], y] -
                                                        sigmaxy_gpu[x + idx_fd_gpu[c, 2], y]) * one_dx
                        dsigmayy_dy += coefs_gpu[c] * (sigmayy_gpu[x, y + idx_fd_gpu[c, 0]] -
                                                        sigmayy_gpu[x, y + idx_fd_gpu[c, 2]]) * one_dy

                    mdsxy_dx_new = b_x_half_gpu[x - offset] * memory_dsigmaxy_dx_gpu[x, y] + a_x_half_gpu[x - offset] * dsigmaxy_dx
                    mdsyy_dy_new = b_y_half_gpu[y - offset] * memory_dsigmayy_dy_gpu[x, y] + a_y_half_gpu[y - offset] * dsigmayy_dy

                    dsigmaxy_dx = dsigmaxy_dx / k_x_half_gpu[x - offset] + mdsxy_dx_new
                    dsigmayy_dy = dsigmayy_dy / k_y_half_gpu[y - offset] + mdsyy_dy_new

                    memory_dsigmaxy_dx_gpu[x, y] = mdsxy_dx_new
                    memory_dsigmayy_dy_gpu[x, y] = mdsyy_dy_new

                    if rho4 > 0.0:
                        # Adiciona o sinal de fonte, se o pixel fizer parte de uma fonte
                        idx_src  = idx_src_gpu[x, y]
                        val_src  = 0.0
                        if idx_src != -1:
                            val_src = source_term_gpu[it - 1, idx_src] * dt / rho4

                        vy_gpu[x, y] += val_src + (dsigmaxy_dx + dsigmayy_dy) * dt / rho4
                else:
                    # Condicao de Dirichlet
                    vy_gpu[x, y] = 0.0

                # Norma L2 maxima (max sobre todos os pontos)
                v2 = vx_gpu[x, y] * vx_gpu[x, y] + vy_gpu[x, y] * vy_gpu[x, y]
                ti.atomic_max(v_l2_norm_gpu[None], v2)

        # ============================================================
        # Laco principal de tempo
        # ============================================================
        visco_flag = 1 if visco_on else 0

        t_gpu = time()
        for it in range(1, self._n_steps + 1):
            # Zera a norma L2 antes de cada passo
            v_l2_norm_gpu[None] = 0.0

            # Calculo das tensoes
            sigma_kernel(dt, one_dx, one_dy, nx, ny, ord, it, visco_flag)

            # Calculo das velocidades
            velocity_kernel(dt, one_dx, one_dy, nx, ny, ord, it)

            vsn2 = float(np.sqrt(v_l2_norm_gpu[None]))

            if (it % self._it_display) == 0 or it == 5:
                if self._show_debug:
                    print(f'Time step # {it} out of {self._n_steps}')
                    print(f'Max norm velocity vector V (m/s) = {vsn2}')

                if self._show_anim:
                    self._windows_gpu[0].imv.setImage(
                        sigmayy_gpu.to_numpy()[ix_min:ix_max, iy_min:iy_max],
                        levels=[self._min_val_fields, self._max_val_fields]
                    )
                    self._app.processEvents()

            # Verificacao de estabilidade
            if vsn2 > STABILITY_THRESHOLD:
                raise StabilityError("Simulacao tornando-se instavel", vsn2)

        sim_time = time() - t_gpu

        # ============================================================
        # Coleta dos resultados finais
        # ============================================================
        sigmayy_result      = sigmayy_gpu.to_numpy()
        sens_sigmayy_result = sens_sigyy_gpu.to_numpy()

        return {
            "stress":      sigmayy_result,
            "sens_stress": sens_sigmayy_result,
            "gpu_str":     ti.lang.impl.current_cfg().arch.name,
            "sim_time":    sim_time
        }


# ----------------------------------------------------------
# Avaliacao dos parametros na linha de comando
# ----------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', help='Configuration file', default='config.json')
args = parser.parse_args()

# Cria a instancia do simulador
sim_instance = SimulatorTaichiViscoelastic(args.config)

# Executa simulacao
try:
    sim_instance.run()

except KeyError as key:
    print(f"Chave {key} nao encontrada no arquivo de configuracao.")

except ValueError as value:
    print(value)