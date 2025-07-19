# =======================
# Importacao de pacotes de uso geral
# =======================
# import argparse
# from time import time
import numpy as np
# from sim_support import *
from sim_support.simulator import Simulator

# ======================
# Importacao de pacotes especificos para a implementacao do simulador
# ======================
import taichi as ti
from findiff import coefficients as fdcoeffs

# -----------------------------------------------------------------------------
# Aqui deve ser implementado o simulador como uma classe herdada de Simulator
# -----------------------------------------------------------------------------

@ti.data_oriented
class SimulatorTaichi(Simulator):
    def __init__(self, file_config):
        # Chama do construtor padrao, que le o arquivo de configuracao
        super().__init__(file_config)

        # Define o nome do simulador
        self._name = "Taichi"

        self._npFtype = np.float32
        self._tiFtype = ti.float32

        try: self._Nxyz = (self._nx,); self._Nxyz += (self._ny,); self._Nxyz += (self._nz,)
        except AttributeError: pass

        try: self._xyz_s = (self._ix_src,); self._xyz_s += (self._iy_src,); self._xyz_s += (self._iz_src,)
        except AttributeError: pass

        try: self._xyz_r = (self._ix_rec,); self._xyz_r += (self._iy_rec,); self._xyz_r += (self._iz_rec,)
        except AttributeError: pass

        self._xyz_s = tuple(tuple(i) for i in np.array(self._xyz_s).T.astype(self._npFtype))  # Coordinates of sources
        self._xyz_r = tuple(tuple(i) for i in np.array(self._xyz_r).T.astype(self._npFtype))  # Coordinates of receivers

        offsets = tuple(np.arange(-self._deriv_acc, self._deriv_acc) + .5)
        self._c = tuple(fdcoeffs(deriv=1, offsets=offsets)["coefficients"][self._deriv_acc:].astype(self._npFtype))

        ti.init(arch=ti.gpu)

        self._c2 = ti.field(self._tiFtype, shape=self._Nxyz)
        self.zero_boundaries(self._c2)

        if self._source_term.ndim == 1:
            self._source_term = self._source_term[np.newaxis]
        self._source = [ti.field(self._tiFtype, shape=self._n_steps) for _ in range(self._n_src)]
        for ns in range(self._n_src):
            self._source[ns].from_numpy((self._dt**2 * self._source_term[ns]).astype(self._npFtype))
        #self._receiver = [ti.field(self._tiFtype, shape=self._n_steps) for _ in range(self._n_rec)]
        self._receiver = ti.field(self._tiFtype, shape=(self._n_steps, self._n_rec))

        # ABC
        bx = np.zeros((self._Nxyz[0], 1), dtype=self._npFtype)
        by = np.zeros((1, self._Nxyz[1]), dtype=self._npFtype)
        bx[1:-1, :] = self._b_x
        by[:, 1:-1] = self._b_y
        b = [None, None]
        b[0] = (bx @ np.ones(self._Nxyz[1])[np.newaxis]).astype(self._npFtype)
        b[1] = (np.ones(self._Nxyz[0])[:, np.newaxis] @ by).astype(self._npFtype)
        self._b = [ti.field(self._tiFtype, shape=self._Nxyz) for _ in range(len(self._Nxyz))]
        for nd in range(len(self._Nxyz)):
            self._b[nd].from_numpy(b[nd])

        # Definicao dos limites para a plotagem dos campos
        self._v_max = 1.0
        self._v_min = - self._v_max

    @ti.func
    def _D(self, nd, u, xyz, bf):
        d = 0.
        for nc in ti.static(range(len(self._c))):
            # # Solution 1
            # xyz_p = xyz[:]
            # xyz_m = xyz[:]
            # xyz_p[nd] += nc + bf
            # xyz_m[nd] -= nc - bf + 1
            # d += ti.static(fdstg[nc]) * (u[xyz_p] - u[xyz_m])

            # # Solution 2
            # xyz[nd] += nc + bf
            # a = u[xyz]
            # xyz[nd] += - 2 * nc - 1
            # d += ti.static(fdstg[nc]) * (a - u[xyz])
            # xyz[nd] += nc + 1 - bf

            # Solution 3
            xyz_tmp = xyz[:]
            xyz_tmp[nd] += nc + bf
            a = u[xyz_tmp]
            xyz_tmp[nd] += - 2 * nc - 1
            d += ti.static(self._c[nc]) * (a - u[xyz_tmp])
        return d

    @ti.kernel
    def zero_boundaries(self, prmtr: ti.template()):
        for xyz in ti.grouped(prmtr):
            cond = False
            for nd in ti.static(range(len(self._Nxyz))):
                cond = cond or xyz[nd] < self._deriv_acc or xyz[nd] >= self._Nxyz[nd] - self._deriv_acc
            if cond:
                prmtr[xyz] = 0.

    def _show_anim_func(self, nt, u):
        if not nt % self._it_display:
            u_np = u.to_numpy()[self._roi.get_ix_min():self._roi.get_ix_max(), self._roi.get_iy_min():self._roi.get_iy_max()]
            self._windows_gpu[0].imv.setImage(u_np, levels=[self._v_min, self._v_max])
            self._app.processEvents()

#     def implementation(self):
#         super().implementation()
#
#         # --------------------------------------------
#         # Aqui comeca o codigo especifico do simulador
#         # --------------------------------------------
#
#
#
#         # "vx": vx, "vy": vy, "sens_vx": sens_vx, "sens_vy": sens_vy
#         # return {"sim_time": sim_time, "gpu_str": str(ti.lang.impl.current_cfg().arch),
#         #         "sens_pressure": receiver.to_numpy(), "pressure": p_0.to_numpy()}
#         return {"sim_time": 0., "gpu_str": str(ti.lang.impl.current_cfg().arch),
#                 "sens_pressure": None, "pressure": None}
#
# # ----------------------------------------------------------
# # Avaliacao dos parametros na linha de comando
# # ----------------------------------------------------------
# parser = argparse.ArgumentParser()
# # parser.add_argument('-c', '--config', help='Configuration file', default='config.json')
# default_config_file = "ensaios/ponto/ponto.json"
# parser.add_argument('-c', '--config', help='Configuration file', default=default_config_file)
# args = parser.parse_args()
#
# # Cria a instancia do simulador
# sim_instance = SimulatorTaichi(args.config)
#
# #%% Executa simulacao
# try:
#     sim_instance.run()
#     # pass
#
# except KeyError as key:
#     print(f"Chave {key} nao encontrada no arquivo de configuracao.")
#
# except ValueError as value:
#     print(value)