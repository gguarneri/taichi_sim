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
import findiff
from scipy.ndimage import correlate

# -----------------------------------------------------------------------------
# Aqui deve ser implementado o simulador como uma classe herdada de Simulator
# -----------------------------------------------------------------------------
class SimulatorCPUNumpy(Simulator):
    def __init__(self, file_config):
        # Chama do construtor padrao, que le o arquivo de configuracao
        super().__init__(file_config)

        # Define o nome do simulador
        self._name = "CPU Numpy"

    def implementation(self):
        super().implementation()

        # --------------------------------------------
        # Aqui comeca o codigo especifico do simulador
        # --------------------------------------------
        # ti.init(arch=ti.gpu)
        # tiFtype = ti.float32
        npFtype = np.float32

        try:
            Nxyz = self._nx, self._ny, self._nz
            xyz_s = self._ix_src, self._iy_src, self._iz_src
            xyz_r = self._ix_rec, self._iy_rec, self._iz_rec
            Nxyz_abc = (((self._roi._pml_xmin_len, self._ny, self._nz), (self._roi._pml_xmax_len, self._ny, self._nz)),
                        ((self._nx, self._roi._pml_ymin_len, self._nz), (self._nx, self._roi._pml_ymax_len, self._nz)),
                        ((self._nx, self._ny, self._roi._pml_ymin_len), (self._nx, self._ny, self._roi._pml_ymax_len)))

        except AttributeError:
            Nxyz = self._nx, self._ny
            xyz_s = self._ix_src, self._iy_src
            xyz_r = self._ix_rec, self._iy_rec
            Nxyz_abc = (((self._roi._pml_xmin_len, self._ny), (self._roi._pml_xmax_len, self._ny)),
                        ((self._nx, self._roi._pml_ymin_len), (self._nx, self._roi._pml_ymax_len)))

        # Nxyz_abc = (((18, 816), (18, 816)), ((816, 0), (816, 0)))

        psi_p = [[np.zeros(lr, dtype=npFtype) for lr in nxyz] for nxyz in Nxyz_abc]

        xyz_s = tuple(tuple(i) for i in np.array(xyz_s).T)  # Coordinates of sources
        xyz_r = tuple(tuple(i) for i in np.array(xyz_r).T)  # Coordinates of receivers

        Nd = len(Nxyz)  # Number of dimensions
        Ns = len(xyz_s)  # Number of sources
        Nr = len(xyz_r)  # Number of receivers

        c2 = np.zeros(Nxyz, dtype=npFtype)
        p = np.zeros(Nxyz, dtype=npFtype)
        v = [np.zeros(Nxyz, dtype=npFtype) for _ in range(Nd)]
        source = np.zeros(self._n_steps, dtype=npFtype)
        receiver = np.zeros((self._n_steps, self._n_rec), dtype=npFtype)

        self._deriv_acc = 2
        offset = (np.concatenate((np.arange(-self._deriv_acc, 0), np.arange(self._deriv_acc))) + .5).astype(npFtype)
        # fd = findiff.coefficients(deriv=1, offsets=list(offset))["coefficients"][round(self._deriv_acc):].astype(npFtype)
        fd = findiff.coefficients(deriv=1, offsets=list(offset))["coefficients"].astype(npFtype)
        Nc = len(fd)
        #source = np.cumsum(self._source_term * self._dt).astype(npFtype)
        source = (self._source_term * self._dt).astype(npFtype)

        c2.fill(self._cp ** 2 * self._dt / self._dx)
        # rho_.fill(self._dt / (self._rho * self._dx))
        # K.fill(self._dt / (self._cp ** 2 * self._rho * self._dx))
        dtOdx = self._dt / self._dx

        # @ti.kernel
        # def parameters_zero_boundaries():
        #     for xyz in ti.grouped(c2):
        #         cond = False
        #         for k in ti.static(range(Nd)):
        #             cond = cond or xyz[k] < self._deriv_acc - 1 or xyz[k] > Nxyz[k] - self._deriv_acc
        #         if cond:
        #             # rho_[xyz] = 0.
        #             # K[xyz] = 0.
        #             c2[xyz] = 0.
        #
        # parameters_zero_boundaries()

        # @ti.func
        # def lap(u, xyz):
        #     lp = ti.static(Nd) * ti.static(fd[0]) * u[xyz]
        #     for nc in ti.static(range(1, Nc)):  # compile-time loop unrolling
        #         for nd in ti.static(range(Nd)):
        #             xyz[nd] += nc
        #             a = u[xyz]
        #             xyz[nd] -= 2 * nc
        #             lp += ti.static(fd[nc]) * (a + u[xyz])
        #             xyz[nd] += nc
        #     return lp

        # @ti.func
        # def div(u, xyz):
        #     d = 0.
        #     for nc in ti.static(range(Nc)):
        #         for nd in ti.static(range(Nd)):
        #             xyz_p = xyz
        #             xyz_m = xyz
        #             xyz_p[nd] += nc + 1
        #             xyz_m[nd] -= nc
        #             d += ti.static(fd[nc]) * (u[nd, xyz_p] - u[nd, xyz_m])
        #     return d

        # @ti.func
        def D_(nd, u, bf):
            d = np.zeros(Nxyz, dtype=npFtype)
            for xyz in np.ndindex(Nxyz):
                for nc in range(Nc):
                    xyz_p = list(xyz)
                    xyz_m = list(xyz)
                    xyz_p[nd] += nc + bf
                    xyz_m[nd] -= nc + 1 + bf
                    d[xyz] += fd[nc] * (u[tuple(xyz_p)] - u[tuple(xyz_m)])
            return d

        # @nb.njit
        def D_(dim, u, bf):
            w = u.swapaxes(0, dim)
            y = np.zeros((w.shape[0] + 1, w.shape[1]), dtype=npFtype)
            for i, c in enumerate(fd):
                y[:-i - 1, ...] += c * w[i:, ...]
                y[i + 1:, ...] -= c * w[:w.shape[0] - i, ...]
            return y[bf:y.shape[0] + bf - 1, ...].swapaxes(0, dim)

        def D(dim, u, bf):
            return correlate(u, fd, mode="constant", cval=0, origin=-bf, axes=dim)

        #
        # @ti.kernel
        def update_p(p, v, nt):
            for nd in range(Nd):
                p -= c2 * D(nd, v[nd], 1)

            for ns in range(Ns):
                #if (xyz == xyz_s[ns]).all():
                p[xyz_s[ns]] += source[nt]
            for nr in range(Nr):
                # if (xyz == xyz_r[nr]).all():
                receiver[nt, nr] = p[xyz_r[nr]]
            # p[xyz_s] += source[nt]
        #    for xyz in ti.grouped(p):
        #        # p[xyz] -= K[xyz] * div(v, xyz)
        #         for nd in ti.static(range(Nd)):
        #             p[xyz] -= c2[xyz] * D(nd, v[nd], xyz, 1)
        #         for ns in ti.static(range(Ns)):
        #             if (xyz == xyz_s[ns]).all():
        #                 p[xyz] += source[nt]
        #         for nr in ti.static(range(Nr)):
        #             if (xyz == xyz_r[nr]).all():
        #                 receiver[nt, nr] = p[xyz]
        #
        # @ti.kernel
        # def update_v():
        #     for xyz in ti.grouped(p):
        #         for nd in ti.static(range(Nd)):
        #             #v[nd, xyz] -= rho_[xyz] * D(nd, p, xyz)
        #             v[nd][xyz] -= dtOdx * D(nd, p, xyz, 0)

        def update_v(p, v):
            for nd in range(Nd):
                v[nd] -= dtOdx * D(nd, p, 0)

        # @ti.kernel
        # def circulate_buffers():
        #     for xyz in ti.grouped(u_0):
        #         u_2[xyz] = u_1[xyz]
        #         u_1[xyz] = u_0[xyz]

        # view = 10
        # vmm = 1e6
        # cmap = mpl.colormaps['bwr']
        # norm = lambda x: .5 + x / (2 * vmm)
        # window = ti.ui.Window(name="Wave", res=Nxyz[0:2], pos=(0, 0))
        # canvas = window.get_canvas()
        # Definicao dos limites para a plotagem dos campos
        v_max = 100.0
        v_min = - v_max
        ix_min = self._roi.get_ix_min()
        ix_max = self._roi.get_ix_max()
        iy_min = self._roi.get_iz_min()
        iy_max = self._roi.get_iz_max()
        t_init = time()
        for nt in range(self._n_steps):
            update_p(p, v, nt)
            update_v(p, v)
            if self._show_anim:
                if not nt % self._it_display:
                    self._windows_gpu[0].imv.setImage(p[ix_min:ix_max, iy_min:iy_max], levels=[v_min, v_max])
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
        return {"sim_time": sim_time, "gpu_str": "CPU",
                "sens_pressure": receiver, "pressure": p}

# ----------------------------------------------------------
# Avaliacao dos parametros na linha de comando
# ----------------------------------------------------------
parser = argparse.ArgumentParser()
default_config_file = "ensaios/ponto/ponto.json"
parser.add_argument('-c', '--config', help='Configuration file', default=default_config_file)
args = parser.parse_args()

# Cria a instancia do simulador
sim_instance = SimulatorCPUNumpy(args.config)

#%% Executa simulacao
try:
    sim_instance.run()
    # pass

except KeyError as key:
    print(f"Chave {key} nao encontrada no arquivo de configuracao.")

except ValueError as value:
    print(value)