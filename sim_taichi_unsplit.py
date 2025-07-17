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
class SimulatorTaichiUnsplit(Simulator):
    def __init__(self, file_config):
        # Chama do construtor padrao, que le o arquivo de configuracao
        super().__init__(file_config)

        # Define o nome do simulador
        self._name = "Taichi Unsplit"

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
            Nxyz_abc = (((self._roi._pml_xmin_len, self._ny, self._nz), (self._roi._pml_xmax_len, self._ny, self._nz)),
                        ((self._nx, self._roi._pml_ymin_len, self._nz), (self._nx, self._roi._pml_ymax_len, self._nz)),
                        ((self._nx, self._ny, self._roi._pml_ymin_len), (self._nx, self._ny, self._roi._pml_ymax_len)))

        except AttributeError:
            Nxyz = self._nx, self._ny
            xyz_s = self._ix_src, self._iy_src
            xyz_r = self._ix_rec, self._iy_rec
            Nxyz_abc = (((self._roi._pml_xmin_len, self._ny), (self._roi._pml_xmax_len, self._ny)),
                        ((self._nx, self._roi._pml_ymin_len), (self._nx, self._roi._pml_ymax_len)))

        xyz_s = tuple(tuple(i) for i in np.array(xyz_s).T)  # Coordinates of sources
        xyz_r = tuple(tuple(i) for i in np.array(xyz_r).T)  # Coordinates of receivers

        Nd = len(Nxyz)  # Number of dimensions
        Ns = len(xyz_s)  # Number of sources
        Nr = len(xyz_r)  # Number of receivers

        c2 = ti.field(tiFtype, shape=Nxyz)
        p_0 = ti.field(tiFtype, shape=Nxyz)
        p_1 = ti.field(tiFtype, shape=Nxyz)
        p_2 = ti.field(tiFtype, shape=Nxyz)
        source = ti.field(tiFtype, shape=self._n_steps)
        source.from_numpy((self._dt**2 * self._source_term).astype(npFtype))

        receiver = ti.field(tiFtype, shape=(self._n_steps, self._n_rec))

        offsets = tuple(np.arange(-self._deriv_acc//2, self._deriv_acc//2 + 1))
        fd_np = fdcoeffs(deriv=2, offsets=offsets)["coefficients"][self._deriv_acc//2:]
        fd = tuple(fd_np.astype(npFtype))
        Nc = len(fd)

        p_0.fill(0.)
        p_1.fill(0.)
        p_2.fill(0.)
        c2.fill(self._cp**2 * self._dt**2 / self._dx**2)

        @ti.kernel
        def c2_zero_boundaries():
            for xyz in ti.grouped(c2):
                cond = False
                for nd in ti.static(range(Nd)):
                    cond = cond or xyz[nd] < self._deriv_acc - 1 or xyz[nd] > Nxyz[nd] - self._deriv_acc
                if cond:
                    c2[xyz] = 0.

        c2_zero_boundaries()

        @ti.func
        def lap(u, xyz):
            lp = ti.static(Nd) * ti.static(fd[0]) * u[xyz]
            for nc in ti.static(range(1, Nc)):  # compile-time loop unrolling
                for nd in ti.static(range(Nd)):
                    xyz[nd] += nc
                    a = u[xyz]
                    xyz[nd] -= 2 * nc
                    lp += ti.static(fd[nc]) * (a + u[xyz])
                    xyz[nd] += nc
            return lp

        # @ti.func
        # def equal_coords_(c1, c2):
        #     for nd in ti.static(range(Nd)):
        #         if c1[nd] != c2[nd]:
        #             return False
        #     return True

        # def equal_coords(c1, c2):
        #     for nd in range(Nd):
        #         if c1[nd] != c2[nd]:
        #             return False
        #     return True

        @ti.kernel
        def update_p(nt: int):
            for xyz in ti.grouped(p_0):
                p_0[xyz] = 2 * p_1[xyz] - p_2[xyz] + c2[xyz] * lap(p_1, xyz)
                abc = False
                for nd in ti.static(range(Nd)):
                    abc = abc or xyz[nd] < Nxyz_abc[nd][0][nd] or xyz[nd] > Nxyz[nd] - Nxyz_abc[nd][1][nd]
                if abc:
                    p_0[xyz] += .001 * nt
                for ns in ti.static(range(Ns)):
                    if all(xyz == xyz_s[ns]):
                        p_0[xyz] += source[nt]
                for nr in ti.static(range(Nr)):
                    if all(xyz == xyz_r[nr]):
                        receiver[nt, nr] = p_0[xyz]

        @ti.kernel
        def circulate_buffers():
            for xyz in ti.grouped(p_0):
                p_2[xyz] = p_1[xyz]
                p_1[xyz] = p_0[xyz]

        # Definicao dos limites para a plotagem dos campos
        v_max = 1.0
        v_min = - v_max
        ix_min = self._roi.get_ix_min()
        ix_max = self._roi.get_ix_max()
        iy_min = self._roi.get_iz_min()
        iy_max = self._roi.get_iz_max()
        t_init = time()
        for nt in range(self._n_steps):
            update_p(nt)
            circulate_buffers()
            if self._show_anim:
                if not nt % self._it_display:
                    u_0np = p_0.to_numpy()[ix_min:ix_max, iy_min:iy_max]
                    self._windows_gpu[0].imv.setImage(u_0np, levels=[v_min, v_max])
                    self._app.processEvents()

        sim_time = time() - t_init

        # "vx": vx, "vy": vy, "sens_vx": sens_vx, "sens_vy": sens_vy
        return {"sim_time": sim_time, "gpu_str": str(ti.lang.impl.current_cfg().arch),
                "sens_pressure": receiver.to_numpy(), "pressure": p_0.to_numpy()}

# ----------------------------------------------------------
# Avaliacao dos parametros na linha de comando
# ----------------------------------------------------------
parser = argparse.ArgumentParser()
# parser.add_argument('-c', '--config', help='Configuration file', default='config.json')
default_config_file = "ensaios/ponto/ponto.json"
parser.add_argument('-c', '--config', help='Configuration file', default=default_config_file)
args = parser.parse_args()

# Cria a instancia do simulador
sim_instance = SimulatorTaichiUnsplit(args.config)

#%% Executa simulacao
try:
    sim_instance.run()

except KeyError as key:
    print(f"Chave {key} nao encontrada no arquivo de configuracao.")

except ValueError as value:
    print(value)