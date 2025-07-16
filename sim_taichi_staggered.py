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
from findiff import coefficients as fdcoeffs

# -----------------------------------------------------------------------------
# Aqui deve ser implementado o simulador como uma classe herdada de Simulator
# -----------------------------------------------------------------------------
class SimulatorTaichiStaggered(Simulator):
    def __init__(self, file_config):
        # Chama do construtor padrao, que le o arquivo de configuracao
        super().__init__(file_config)

        # Define o nome do simulador
        self._name = "Taichi Staggered"

    def implementation(self):
        super().implementation()

        # --------------------------------------------
        # Aqui comeca o codigo especifico do simulador
        # --------------------------------------------
        ti.init(arch=ti.gpu)
        tiFtype = ti.float32
        npFtype = np.float32

        try:
            Nxyz = self._nx, self._ny, self._nz
            xyz_s = self._ix_src, self._iy_src, self._iz_src
            xyz_r = self._ix_rec, self._iy_rec, self._iz_rec
        except AttributeError:
            Nxyz = self._nx, self._ny
            xyz_s = self._ix_src, self._iy_src
            xyz_r = self._ix_rec, self._iy_rec

        xyz_s = tuple(tuple(i) for i in np.array(xyz_s).T)  # Coordinates of sources
        xyz_r = tuple(tuple(i) for i in np.array(xyz_r).T)  # Coordinates of receivers

        Nd = len(Nxyz)  # Number of dimensions
        Ns = len(xyz_s)  # Number of sources
        Nr = len(xyz_r)  # Number of receivers

        c2 = ti.field(tiFtype, shape=Nxyz)
        # rho_ = ti.field(tiFtype, shape=Nxyz)
        # K = ti.field(tiFtype, shape=Nxyz)
        p = ti.field(tiFtype, shape=Nxyz)
        v = [ti.field(tiFtype, shape=Nxyz) for _ in range(Nd)]
        #u_2 = ti.field(tiFtype, shape=Nxyz)
        source = ti.field(tiFtype, shape=self._n_steps)
        source.from_numpy((self._dt * np.cumsum(self._source_term * self._dt)).astype(npFtype))
        # source.from_numpy((self._source_term * self._dt).astype(npFtype))

        receiver = ti.field(tiFtype, shape=(self._n_steps, self._n_rec))

        offsets = tuple(np.arange(-self._deriv_acc, self._deriv_acc) + .5)
        fd_np = fdcoeffs(deriv=1, offsets=offsets)["coefficients"][self._deriv_acc:]
        fd = tuple(fd_np.astype(npFtype))
        Nc = len(fd)

        p.fill(0.)
        for nd in range(Nd):
            v[nd].fill(0.)
        c2.fill(self._cp ** 2 * self._dt / self._dx)
        # rho_.fill(self._dt / (self._rho * self._dx))
        # K.fill(self._dt / (self._cp ** 2 * self._rho * self._dx))
        dtOdx = self._dt / self._dx

        @ti.kernel
        def parameters_zero_boundaries():
            for xyz in ti.grouped(c2):
                cond = False
                for nd in ti.static(range(Nd)):
                    cond = cond or xyz[nd] < self._deriv_acc or xyz[nd] >= Nxyz[nd] - self._deriv_acc
                if cond:
                    # rho_[xyz] = 0.
                    # K[xyz] = 0.
                    c2[xyz] = 0.

        parameters_zero_boundaries()

        @ti.func
        def D(nd, u, xyz, bf):
            d = 0.
            for nc in ti.static(range(Nc)):
                # # Solution 1
                # xyz_p = xyz[:]
                # xyz_m = xyz[:]
                # xyz_p[nd] += nc + bf
                # xyz_m[nd] -= nc - bf + 1
                # d += ti.static(fd[nc]) * (u[xyz_p] - u[xyz_m])

                # Solution 2
                xyz[nd] += nc + bf
                a = u[xyz]
                xyz[nd] += - 2 * nc - 1
                d += ti.static(fd[nc]) * (a - u[xyz])
                xyz[nd] += nc + 1 - bf
            return d

        @ti.kernel
        def update_p(nt: int):
            for xyz in ti.grouped(p):
                for nd in ti.static(range(Nd)):
                    p[xyz] -= c2[xyz] * D(nd, v[nd], xyz, 1)
                for ns in ti.static(range(Ns)):
                    if (xyz == xyz_s[ns]).all():
                        p[xyz] += source[nt]
                for nr in ti.static(range(Nr)):
                    if (xyz == xyz_r[nr]).all():
                        receiver[nt, nr] = p[xyz]

        @ti.kernel
        def update_v():
            for xyz in ti.grouped(p):
                for nd in ti.static(range(Nd)):
                    v[nd][xyz] -= dtOdx * D(nd, p, xyz, 0)

        # view = 10
        # vmm = 1e6
        # cmap = mpl.colormaps['bwr']
        # norm = lambda x: .5 + x / (2 * vmm)
        # window = ti.ui.Window(name="Wave", res=Nxyz[0:2], pos=(0, 0))
        # canvas = window.get_canvas()
        # Definicao dos limites para a plotagem dos campos
        v_max = 1.0
        v_min = - v_max
        ix_min = self._roi.get_ix_min()
        ix_max = self._roi.get_ix_max()
        iy_min = self._roi.get_iz_min()
        iy_max = self._roi.get_iz_max()
        t_init = time()
        for nt in range(self._n_steps):
            update_v()
            update_p(nt)
            if self._show_anim:
                if not nt % self._it_display:
                    p_np = p.to_numpy()
                    self._windows_gpu[0].imv.setImage(p_np[ix_min:ix_max, iy_min:iy_max], levels=[v_min, v_max])
                    self._app.processEvents()
            # if view:
            #     if not nt % view:
            #         u0np = u_0.to_numpy()
            #         if u0np.ndim > 2:
            #             u0np = u0np[..., Nxyz[-1] // 2]
            #         u0npcmap = cmap(norm(u0np)).astype(np.float32)
            #         canvas.set_image(u0npcmap)
            #         window.show()

        sim_time = time() - t_init

        #return {"vx": vx, "vy": vy, "pressure": pressure,
        #        "sens_vx": sens_vx, "sens_vy": sens_vy, "sens_pressure": sens_pressure,
        #        "gpu_str": self._device.adapter.info["device"], "sim_time": sim_time}
        return {"sim_time": sim_time, "gpu_str": str(ti.lang.impl.current_cfg().arch),
                "sens_pressure": receiver.to_numpy(), "pressure": p.to_numpy()}

# ----------------------------------------------------------
# Avaliacao dos parametros na linha de comando
# ----------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', help='Configuration file', default='config.json')
args = parser.parse_args()

# Cria a instancia do simulador
sim_instance = SimulatorTaichiStaggered(args.config)

#%% Executa simulacao
try:
    sim_instance.run()

except KeyError as key:
    print(f"Chave {key} nao encontrada no arquivo de configuracao.")

except ValueError as value:
    print(value)