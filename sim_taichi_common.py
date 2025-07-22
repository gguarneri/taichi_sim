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

        ti.init(arch=ti.gpu)

        self._npItype = np.int32
        self._npFtype = np.float32
        self._tiItype = ti.int32
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

        self._c2 = ti.field(self._tiFtype, shape=self._Nxyz)
        #self._c2.fill(self._cp**2 * self._dt**2 / self._dx**2)
        self._c2.fill(self._cp**2)
        self._zero_boundaries(self._c2)

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
        try:  # TODO: reavaliar xyz
            self._Nabc = ((self._roi._pml_xmin_len, self._roi._pml_xmax_len),)
            self._Nabc += ((self._roi._pml_zmin_len, self._roi._pml_zmax_len),)
            self._Nabc += ((self._roi._pml_ymin_len, self._roi._pml_ymax_len),)
        except AttributeError: pass

        self._b_field = ti.field(self._tiFtype, shape=np.max(self._Nabc))
        self._b_field.from_numpy(self._b_x[:self._Nabc[0][0], 0])

        # Definicao dos limites para a plotagem dos campos
        self._v_max = 10_000.
        self._v_min = - self._v_max

    # @ti.func
    # def _b_func(self, x, nxyz, nabc):
    #     i, m = 0, 0
    #     if x < nabc[0]:
    #         i, m = x, 1
    #     elif x >= nxyz - nabc[1]:
    #         i, m = nxyz - x - 1, 1
    #     return 1 + m * (self._b_field[i] - 1)

    @ti.func
    def _update_abc_(self, D, psi, xyz, nxyz, nabc, nd):
        i, m = 0, 0
        if xyz[nd] < nabc[0]:
            i, m = xyz[nd], 1
        elif xyz[nd] >= nxyz - nabc[1]:
            i, m = nxyz - xyz[nd] - 1, 1
        if m:
            psi[xyz] = self._b_field[i] * psi[xyz] + (self._b_field[i] - 1) * D

    @ti.func
    def _update_abc(self, D, psi, xyz, nxyz, nabc, nd):
        r = D
        if xyz[nd] < nabc[0]:
            r += psi[xyz]
            psi[xyz] = self._b_field[xyz[nd]] * psi[xyz] + (self._b_field[xyz[nd]] - 1) * D
        elif xyz[nd] >= nxyz - nabc[1]:
            r += psi[xyz]
            psi[xyz] = self._b_field[nxyz - xyz[nd] - 1] * psi[xyz] + (self._b_field[nxyz - xyz[nd] - 1] - 1) * D
        return r

    @ti.func
    def _a_func(self, nd, xyz):
        return self._b[nd][xyz] - 1

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
    def _zero_boundaries(self, prmtr: ti.template()):
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

if __name__ == '__main__':
    import argparse
    import matplotlib.pyplot as plt
    # import matplotlib as mpl
    # mpl.use('Qt5Agg')
    # plt.ion()

    parser = argparse.ArgumentParser()
    default_config_file = "ensaios/ponto/ponto.json"
    parser.add_argument('-c', '--config', help='Configuration file', default=default_config_file)
    args = parser.parse_args()
    sim = SimulatorTaichiCommon(args.config)

    plt.imshow(sim._b[0].to_numpy())