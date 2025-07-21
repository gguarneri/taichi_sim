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
class SimulatorTaichiCommon(Simulator):
    def __init__(self, file_config):
        # Chama do construtor padrao, que le o arquivo de configuracao
        super().__init__(file_config)

        # Define o nome do simulador
        self._name = "Taichi Common"

        self._npFtype = np.float32
        self._tiFtype = ti.float32

        try: self._Nxyz = (self._nx,); self._Nxyz += (self._ny,); self._Nxyz += (self._nz,)
        except AttributeError: pass

        try: self._xyz_s = (self._ix_src,); self._xyz_s += (self._iy_src,); self._xyz_s += (self._iz_src,)
        except AttributeError: pass

        try: self._xyz_r = (self._ix_rec,); self._xyz_r += (self._iy_rec,); self._xyz_r += (self._iz_rec,)
        except AttributeError: pass

        self._xyz_s = tuple(tuple(i) for i in np.array(self._xyz_s).T)  # Coordinates of sources
        self._xyz_r = tuple(tuple(i) for i in np.array(self._xyz_r).T)  # Coordinates of receivers

        offsets = tuple(np.arange(-self._deriv_acc, self._deriv_acc) + .5)
        self._c = tuple(fdcoeffs(deriv=1, offsets=offsets)["coefficients"][self._deriv_acc:].astype(self._npFtype))

        ti.init(arch=ti.gpu)

        self._c2 = ti.field(self._tiFtype, shape=self._Nxyz)
        #self._c2.fill(self._cp**2 * self._dt**2 / self._dx**2)
        self._c2.fill(self._cp**2)
        self.zero_boundaries(self._c2)

        if self._source_term.ndim == 1:
            self._source_term = self._source_term[np.newaxis]
        self._source_dp = [ti.field(self._tiFtype, shape=self._n_steps) for _ in range(self._n_src)]
        self._source_d2p = [ti.field(self._tiFtype, shape=self._n_steps) for _ in range(self._n_src)]
        for ns in range(self._n_src):
            dp = np.diff(self._source_term[ns], prepend=0)
            self._source_dp[ns].from_numpy(dp.astype(self._npFtype))
            self._source_d2p[ns].from_numpy(np.diff(dp, prepend=0).astype(self._npFtype))
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
        self._v_max = 10_000.
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
            # TODO: reavaliar xyz
            u_np = u.to_numpy()[self._roi.get_ix_min():self._roi.get_ix_max(), self._roi.get_iz_min():self._roi.get_iz_max()]
            self._windows_gpu[0].imv.setImage(u_np, levels=[self._v_min, self._v_max])
            self._app.processEvents()
            # print("Showing animation...", u_np.shape, np.sum(u_np ** 2))

    @ti.func
    def _addSourceD2p(self, p, xyz, nt):
        for ns in ti.static(range(self._n_src)):
            if all(xyz == self._xyz_s[ns]):
                p[xyz] += self._source_d2p[ns][nt]

    @ti.func
    def _addSourceDp(self, p, xyz, nt):
        for ns in ti.static(range(self._n_src)):
            if all(xyz == self._xyz_s[ns]):
                p[xyz] += self._source_dp[ns][nt]

    @ti.func
    def _readSensors(self, p, xyz, nt):
        for nr in ti.static(range(self._n_rec)):
            if all(xyz == self._xyz_r[nr]):
                self._receiver[nt, nr] = p[xyz]