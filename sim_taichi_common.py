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
from collections import namedtuple

# Simple namespace to store positive-negative itens
# pn = namedtuple('pn', ('p', 'n'))

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

        ti.init(arch=ti.gpu, default_fp=ti.f32, default_ip=ti.i32)

        self._npItp = np.int32
        self._npFtp = np.float32

        try: Nxyz = (self._nx,); Nxyz += (self._ny,); Nxyz += (self._nz,)
        except AttributeError: pass
        self._Nd = len(Nxyz)
        self._Nxyz = ti.field(int, self._Nd)
        self._Nxyz.from_numpy(np.array(Nxyz).astype(self._npItp))

        try: self._xyz_s = (self._ix_src,); self._xyz_s += (self._iy_src,); self._xyz_s += (self._iz_src,)
        except AttributeError: pass

        try: self._xyz_r = (self._ix_rec,); self._xyz_r += (self._iy_rec,); self._xyz_r += (self._iz_rec,)
        except AttributeError: pass

        self._xyz_s = tuple(tuple(i) for i in np.array(self._xyz_s).T)  # Coordinates of sources
        self._xyz_r = tuple(tuple(i) for i in np.array(self._xyz_r).T)  # Coordinates of receivers

        offsets = tuple(np.arange(-self._deriv_acc, self._deriv_acc) + .5)
        self._cd1 = tuple(fdcoeffs(deriv=1, offsets=offsets)["coefficients"][self._deriv_acc:].astype(self._npFtp))
        self.offsets = tuple(np.arange(1 - self._deriv_acc, self._deriv_acc))
        self._cd2 = tuple(fdcoeffs(deriv=2, offsets=self.offsets)["coefficients"][round(self._deriv_acc/2):].astype(self._npFtp))

        self._c2 = ti.field(float, self._Nxyz.to_numpy())
        self._c2.fill(self._cp**2)
        self._zero_boundaries(self._c2)

        if self._source_term.ndim == 1:
            self._source_term = self._source_term[np.newaxis]
        self._source_dp = [ti.field(float, self._n_steps) for _ in range(self._n_src)]
        self._source_d2p = [ti.field(float, self._n_steps) for _ in range(self._n_src)]
        for ns in range(self._n_src):
            dp = np.diff(self._source_term[ns], prepend=0)
            self._source_dp[ns].from_numpy(dp.astype(self._npFtp))
            self._source_d2p[ns].from_numpy(np.diff(dp, prepend=0).astype(self._npFtp))
        #self._receiver = [ti.field(float, self._n_steps) for _ in range(self._n_rec)]
        self._receiver = ti.field(float, (self._n_steps, self._n_rec))

        # ABC: Absorbing Boundary Conditions
        try:  # TODO: reavaliar ordem xyz, xzy, etc
            Nabc = ((self._roi._pml_xmax_len, self._roi._pml_xmin_len),)
            Nabc += ((self._roi._pml_zmax_len, self._roi._pml_zmin_len),)
            Nabc += ((self._roi._pml_ymax_len, self._roi._pml_ymin_len),)
        except AttributeError: pass
        self._Nabc = ti.field(int, (3, 2))
        self._Nabc.from_numpy(np.array(Nabc).astype(self._npItp))

        self._b = ti.field(float, np.max(Nabc))
        self._b.from_numpy(self._b_x[:Nabc[0][0], 0])



        # Definicao dos limites para a plotagem dos campos
        self._v_max = 10_000.
        self._v_min = - self._v_max

    @ti.func
    def _pml(self, D, psi, xyz, nd: int):
        r = D
        if xyz[nd] < self._Nabc[nd, 0]:
            r += psi[xyz]
            i = xyz[nd]
            psi[xyz] = self._b[i] * psi[xyz] + (self._b[i] - 1) * D
        elif xyz[nd] > self._Nxyz[nd] - self._Nabc[nd, 1] - 1:
            xyz_r = xyz[:]
            xyz_r[nd] += self._Nabc[nd, 1] - self._Nxyz[nd] + self._Nabc[nd, 0]
            r += psi[xyz_r]
            i = self._Nxyz[nd] - xyz[nd] - 1
            psi[xyz_r] = self._b[i] * psi[xyz_r] + (self._b[i] - 1) * D
        return r

    @ti.func
    def _D(self, u: ti.template(), xyz, nd: int, bf: int, imax: int):  # def _D(self, nd, u, xyz, bf):
        """field, position, dimension, backward or forward"""
        d = 0.
        # iimax = u.shape[nd[0]]
        for nc in ti.static(range(self._deriv_acc)):
            # # Solution 1
            # xyz_p = xyz[:]
            # xyz_n = xyz[:]
            # xyz_p[nd] += nc + bf
            # xyz_n[nd] -= nc - bf + 1
            # a = u[xyz_p] if xyz_p[nd] < imax else 0
            # b = u[xyz_n] if xyz_n[nd] >= 0 else 0
            # d += ti.static(self._cd1[nc]) * (a - b)

            # # Solution 2
            # xyz_tmp = xyz[:]
            # xyz_tmp[nd] += nc + bf
            # a = u[xyz_tmp] if xyz_tmp[nd] < imax else 0
            # xyz_tmp[nd] += - 2 * nc - 1
            # b = u[xyz_tmp] if xyz_tmp[nd] >= 0 else 0
            # d += ti.static(self._cd1[nc]) * (a - b)

            # c = u.shape[0]
            # ti.static_print(nd)
            # c = c[ti.static(nd)]

            # Solution 3
            xyz[nd] += nc + bf
            a = u[xyz] if xyz[nd] < imax else 0
            xyz[nd] += - 2 * nc - 1
            b = u[xyz] if xyz[nd] >= 0 else 0
            xyz[nd] += nc + 1 - bf
            d += ti.static(self._cd1[nc]) * (a - b)

        return d

    @ti.func
    def _lap(self, u, xyz):
        l = ti.static(self._Nd * self._cd2[0]) * u[xyz]
        for nd in ti.static(range(self._Nd)):
            for nc in ti.static(range(1, len(self._cd2))):
                xyz_tmp = xyz[:]
                xyz_tmp[nd] += nc
                a = u[xyz_tmp]
                xyz_tmp[nd] += - 2 * nc
                l += ti.static(self._cd2[nc]) * (a + u[xyz_tmp])
        return l

    @ti.kernel
    def _zero_boundaries(self, prmtr: ti.template()):
        for xyz in ti.grouped(prmtr):
            cond = False
            for nd in ti.static(range(self._Nd)):
                cond = cond or xyz[nd] < self._deriv_acc or xyz[nd] >= self._Nxyz[nd] - self._deriv_acc
            if cond:
                prmtr[xyz] = 0.

    def _show_anim_func(self, nt: int, u):
        if not nt % self._it_display:
            # TODO: reavaliar xyz
            u_np = u.to_numpy()[self._roi.get_ix_min():self._roi.get_ix_max(), self._roi.get_iz_min():self._roi.get_iz_max()]
            self._windows_gpu[0].imv.setImage(u_np, levels=[self._v_min, self._v_max])
            self._app.processEvents()

    @ti.func
    def _addSourceD2p(self, p, xyz, nt: int):
        for ns in ti.static(range(self._n_src)):
            if all(xyz == self._xyz_s[ns]):
                p[xyz] += self._source_d2p[ns][nt]

    @ti.func
    def _addSourceDp(self, p, xyz, nt: int):
        for ns in ti.static(range(self._n_src)):
            if all(xyz == self._xyz_s[ns]):
                p[xyz] += self._source_dp[ns][nt]

    @ti.func
    def _readSensors(self, p, xyz, nt: int):
        for nr in ti.static(range(self._n_rec)):
            if all(xyz == self._xyz_r[nr]):
                self._receiver[nt, nr] = p[xyz]

if __name__ == '__main__':
    import argparse
    # import matplotlib.pyplot as plt
    # import matplotlib as mpl
    # mpl.use('Qt5Agg')
    # plt.ion()

    parser = argparse.ArgumentParser()
    default_config_file = "ensaios/ponto/ponto.json"
    parser.add_argument('-c', '--config', help='Configuration file', default=default_config_file)
    args = parser.parse_args()
    sim = SimulatorTaichiCommon(args.config)

    # print(sim._Nabc)