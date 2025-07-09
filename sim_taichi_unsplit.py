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
import findiff
import matplotlib as mpl
mpl.use('qt5agg')

# -----------------------------------------------------------------------------
# Aqui deve ser implementado o simulador como uma classe herdada de Simulator
# -----------------------------------------------------------------------------
class SimulatorTaichi(Simulator):
    def __init__(self, file_config):
        # Chama do construtor padrao, que le o arquivo de configuracao
        super().__init__(file_config)

        # Define o nome do simulador
        self._name = "Taichi"

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
        u_0 = ti.field(tiFtype, shape=Nxyz)
        u_1 = ti.field(tiFtype, shape=Nxyz)
        u_2 = ti.field(tiFtype, shape=Nxyz)
        source = ti.field(tiFtype, shape=self._n_steps)
        receiver = ti.field(tiFtype, shape=(self._n_steps, self._n_rec))

        fd_np = findiff.coefficients(deriv=2, acc=self._deriv_acc)['center']['coefficients'][self._deriv_acc//2:].astype(npFtype)
        fd = tuple(fd_np)
        Nc = len(fd)
        source.from_numpy(self._source_term.astype(npFtype))

        u_0.fill(0.)
        u_1.fill(0.)
        u_2.fill(0.)
        c2.fill(self._cp**2 * self._dt**2 / self._dx**2)

        @ti.kernel
        def c2_zero_boundaries():
            for xyz in ti.grouped(c2):
                cond = False
                for k in ti.static(range(Nd)):
                    cond = cond or xyz[k] < self._deriv_acc - 1 or xyz[k] > Nxyz[k] - self._deriv_acc
                if cond:
                    c2[xyz] = 0.

        c2_zero_boundaries()

        @ti.func
        def lap(xyz):
            lp = ti.static(Nd) * ti.static(fd[0]) * u_1[xyz]
            for nc in ti.static(range(1, Nc)):  # compile-time loop unrolling
                for k in ti.static(range(Nd)):
                    xyz[k] += nc
                    a = u_1[xyz]
                    xyz[k] -= 2 * nc
                    lp += ti.static(fd[nc]) * (a + u_1[xyz])
                    xyz[k] += nc
            return lp

        @ti.kernel
        def update_fields(nt: int):
            for xyz in ti.grouped(u_0):
                u_0[xyz] = 2. * u_1[xyz] - u_2[xyz] + c2[xyz] * lap(xyz)
                for ns in ti.static(range(Ns)):
                    if (xyz == xyz_s[ns]).all():
                        u_0[xyz] += source[nt]
                for nr in ti.static(range(Nr)):
                    if (xyz == xyz_r[nr]).all():
                        receiver[nt, nr] = u_0[xyz]

        @ti.kernel
        def circulate_buffers():
            for xyz in ti.grouped(u_0):
                u_2[xyz] = u_1[xyz]
                u_1[xyz] = u_0[xyz]

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
            update_fields(nt)
            circulate_buffers()
            if self._show_anim:
                if not nt % self._it_display:
                    u_0np = u_0.to_numpy()
                    self._windows_gpu[0].imv.setImage(u_0np[ix_min:ix_max, iy_min:iy_max], levels=[v_min, v_max])
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
                "sens_pressure": receiver.to_numpy(), "pressure": u_0.to_numpy()}

# ----------------------------------------------------------
# Avaliacao dos parametros na linha de comando
# ----------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', help='Configuration file', default='config.json')
# parser.add_argument('-c', '--config', help='Configuration file', default='/ensaios/ponto/ponto.json')
args = parser.parse_args()

# Cria a instancia do simulador
sim_instance = SimulatorTaichi(args.config)

#%% Executa simulacao
try:
    sim_instance.run()

except KeyError as key:
    print(f"Chave {key} nao encontrada no arquivo de configuracao.")

except ValueError as value:
    print(value)