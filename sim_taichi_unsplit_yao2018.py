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
import math
from findiff import coefficients as fdcoeffs

# -----------------------------------------------------------------------------
# Aqui deve ser implementado o simulador como uma classe herdada de Simulator
# -----------------------------------------------------------------------------

@ti.data_oriented
class SimulatorTaichiUnsplit(Simulator):
    def __init__(self, file_config):
        # Chama do construtor padrao, que le o arquivo de configuracao
        super().__init__(file_config, sim_model="unsplit")

        # Define o nome do simulador
        self._name = "Taichi Unsplit Yao 2018"
        
    def implementation(self):
        super().implementation()

        # --------------------------------------------
        # Aqui comeca o codigo especifico do simulador
        # --------------------------------------------

        ti.init(arch=ti.gpu, default_fp=ti.f32, default_ip=ti.i32)

        # Dimensions
        try: Nxyz_np = (self._nx,); Nxyz_np += (self._ny,); Nxyz_np += (self._nz,)
        except AttributeError: pass
        Nd = len(Nxyz_np)  # Number of dimensions
        Nxyz = ti.field(int, Nd)  # Dimensions in taichi field
        Nxyz.from_numpy(np.array(Nxyz_np).astype(np.int32))
        
        # Absorbing Boundary Conditions (ABC)
        try:  # TODO: reavaliar ordem xyz, xzy, etc
            Nabc_np = ((self._roi._pml_xmax_len, self._roi._pml_xmin_len),)
            Nabc_np += ((self._roi._pml_zmax_len, self._roi._pml_zmin_len),)
            Nabc_np += ((self._roi._pml_ymax_len, self._roi._pml_ymin_len),)
        except AttributeError: pass
        Nabc = ti.field(int, (3, 2))
        Nabc.from_numpy(np.array(Nabc_np).astype(np.int32))

        # Pressure and velocity fields
        p_0 = ti.field(float, shape=Nxyz_np)
        p_1 = ti.field(float, shape=Nxyz_np)
        p_2 = ti.field(float, shape=Nxyz_np)

        # Coordinates of sources
        try: xyz_s = (self._ix_src,); xyz_s += (self._iy_src,); xyz_s += (self._iz_src,)
        except AttributeError: pass

        # Coordinates of receivers
        try: xyz_r = (self._ix_rec,); xyz_r += (self._iy_rec,); xyz_r += (self._iz_rec,)
        except AttributeError: pass

        # Adjusting shape
        xyz_s = tuple(tuple(i) for i in np.array(xyz_s).T)
        xyz_r = tuple(tuple(i) for i in np.array(xyz_r).T)

        # Sources and receivers signals in taichi fields
        if self._source_term.ndim == 1:
            self._source_term = self._source_term[np.newaxis]
        source = [ti.field(float, self._n_steps) for _ in range(self._n_src)]
        for ns in range(self._n_src):
            dp = self._source_term[ns] * self._dt**2 * self._one_dx * self._one_dy
            source[ns].from_numpy(dp.astype(np.float32))
        receiver = ti.field(float, (self._n_steps, self._n_rec))

        c2 = ti.field(float, Nxyz_np)
        #c2.fill(self._cp**2 * self._dt**2 / self._dx**2)
        c2.fill(self._cp**2)

        @ti.func
        def add_source(p, xyz, nt: int):
            for ns in ti.static(range(self._n_src)):
                if all(xyz == xyz_s[ns]):
                    p[xyz] += source[ns][nt]

        @ti.func
        def read_sensors(p, xyz, nt: int):
            for nr in ti.static(range(self._n_rec)):
                if all(xyz == xyz_r[nr]):
                    receiver[nt, nr] = p[xyz]

        @ti.func
        def lap(u, xyz):
            l = ti.static(Nd * self._cd2[0]) * u[xyz]
            for nd in ti.static(range(Nd)):
                for nc in ti.static(range(1, len(self._cd2))):
                    xyz_tmp = xyz[:]
                    xyz_tmp[nd] += nc
                    a = u[xyz_tmp]
                    xyz_tmp[nd] += - 2 * nc
                    l += ti.static(self._cd2[nc]) * (a + u[xyz_tmp])
            return l

        # Absorbing Boundary Conditions (ABC)
        # Effective Absorbing Layer (EAL)
        # Yao et al. 2018
        # 10.1088/1742-2140/aaa4da
        R = .05
        d0_mod = math.log(R) * (3/2) * self._cp
        a_1 = ti.field(float, shape=Nxyz_np)
        a_2 = ti.field(float, shape=Nxyz_np)
        a_3 = ti.field(float, shape=Nxyz_np)

        @ti.kernel
        def fill_abc():
            for xyz in ti.grouped(a_1):
                r = 0.
                for nd in ti.static(range(Nd)):
                    if xyz[nd] < Nabc[nd, 0]:
                        # r = ti.sqrt(r**2 + ((Nabc[nd, 0] - xyz[nd])/Nabc[nd, 0])**2)
                        r = ti.sqrt(r**2 + ((Nabc[nd, 0] - xyz[nd])/Nabc[nd, 0]**(3/2))**2)
                    elif xyz[nd] >= Nxyz[nd] - Nabc[nd, 1]:
                        # r = ti.sqrt(r**2 + ((xyz[nd] - Nxyz[nd] + Nabc[nd, 1] + 1)/Nabc[nd, 1])**2)
                        r = ti.sqrt(r**2 + ((xyz[nd] - Nxyz[nd] + Nabc[nd, 1] + 1)/Nabc[nd, 1]**(3/2))**2)

                d = d0_mod * r**2
                a_1[xyz] = (2 - d**2 * self._dt**2) / (1 - d * self._dt)
                a_2[xyz] = -(1 + d * self._dt) / (1 - d * self._dt)
                a_3[xyz] = c2[xyz] * self._dt**2 / ((1 - d * self._dt) * self._dx**2)

        fill_abc()

        p_0.fill(0.)
        p_1.fill(0.)
        p_2.fill(0.)

        @ti.kernel
        def update_p(nt: int):
            for xyz in ti.grouped(p_0):
                # d = self._d[xyz]
                # oneMd = 1 - d
                # a = (2 - d**2) / oneMd
                # b = - (1 + d) / oneMd
                # c =  dt2Odx2 / oneMd
                # p_0[xyz] = a * p_1[xyz] + b * p_2[xyz] + c * self._c2[xyz] * self._lap(p_1, xyz)

                p_0[xyz] = a_1[xyz] * p_1[xyz] + a_2[xyz] * p_2[xyz] + a_3[xyz] * lap(p_1, xyz)
                add_source(p_0, xyz, nt)
                read_sensors(p_0, xyz, nt)

        @ti.kernel
        def circulate_buffers():
            for xyz in ti.grouped(p_0):
                # Circulate buffers
                p_2[xyz] = p_1[xyz]
                p_1[xyz] = p_0[xyz]

        # Definicao dos limites para a plotagem dos campos
        def show_anim_func(nt: int, u):
            if not nt % self._it_display:
                # TODO: reavaliar xyz
                u_np = u.to_numpy()[
                    self._roi.get_ix_min():self._roi.get_ix_max(), self._roi.get_iz_min():self._roi.get_iz_max()]
                self._windows_gpu[0].imv.setImage(u_np, levels=[self._min_val_fields, self._max_val_fields])
                self._app.processEvents()

        self.offsets = tuple(np.arange(1 - self._deriv_acc, self._deriv_acc))
        self._cd2 = tuple(fdcoeffs(deriv=2, offsets=self.offsets)["coefficients"][round(self._deriv_acc/2):].astype(np.float32))

        t_init = time()
        for nt in range(self._n_steps):
            update_p(nt)
            circulate_buffers()
            if self._show_anim:
                show_anim_func(nt, p_0)
        sim_time = time() - t_init

        # "vx": vx, "vy": vy, "sens_vx": sens_vx, "sens_vy": sens_vy
        return {"sim_time": sim_time, "gpu_str": str(ti.lang.impl.current_cfg().arch),
                "sens_pressure": receiver.to_numpy(), "pressure": p_0.to_numpy()}


# ----------------------------------------------------------
# Avaliacao dos parametros na linha de comando
# ----------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', help='Configuration file', default='config.json')
args = parser.parse_args()

# Cria a instancia do simulador
sim_instance = SimulatorTaichiUnsplit(args.config)

#%% Executa simulacao
try:
    sim_instance.run()
    # pass

except KeyError as key:
    print(f"Chave {key} nao encontrada no arquivo de configuracao.")

except ValueError as value:
    print(value)