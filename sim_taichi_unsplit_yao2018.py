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
from sim_taichi_common import SimulatorTaichiCommon
import math

# -----------------------------------------------------------------------------
# Aqui deve ser implementado o simulador como uma classe herdada de Simulator
# -----------------------------------------------------------------------------

@ti.data_oriented
class SimulatorTaichiUnsplit(SimulatorTaichiCommon):
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

        # Pressure and velocity fields
        p_0 = ti.field(float, shape=self._Nxyz.to_numpy())
        p_1 = ti.field(float, shape=self._Nxyz.to_numpy())
        p_2 = ti.field(float, shape=self._Nxyz.to_numpy())

        # Absorbing Boundary Conditions (ABC)
        # Effective Absorbing Layer (EAL)
        # Yao et al. 2018
        # 10.1088/1742-2140/aaa4da
        R = .05
        d0_mod = math.log(R) * (3/2) * self._cp
        a_1 = ti.field(float, shape=self._Nxyz.to_numpy())
        a_2 = ti.field(float, shape=self._Nxyz.to_numpy())
        a_3 = ti.field(float, shape=self._Nxyz.to_numpy())

        @ti.kernel
        def fill_abc():
            for xyz in ti.grouped(a_1):
                r = 0.
                for nd in ti.static(range(self._Nd)):
                    if xyz[nd] < self._Nabc[nd, 0]:
                        # r = ti.sqrt(r**2 + ((self._Nabc[nd, 0] - xyz[nd])/self._Nabc[nd, 0])**2)
                        r = ti.sqrt(r**2 + ((self._Nabc[nd, 0] - xyz[nd])/self._Nabc[nd, 0]**(3/2))**2)
                    elif xyz[nd] >= self._Nxyz[nd] - self._Nabc[nd, 1]:
                        # r = ti.sqrt(r**2 + ((xyz[nd] - self._Nxyz[nd] + self._Nabc[nd, 1] + 1)/self._Nabc[nd, 1])**2)
                        r = ti.sqrt(r**2 + ((xyz[nd] - self._Nxyz[nd] + self._Nabc[nd, 1] + 1)/self._Nabc[nd, 1]**(3/2))**2)

                d = d0_mod * r**2
                a_1[xyz] = (2 - d**2 * self._dt**2) / (1 - d * self._dt)
                a_2[xyz] = -(1 + d * self._dt) / (1 - d * self._dt)
                a_3[xyz] = self._c2[xyz] * self._dt**2 / ((1 - d * self._dt) * self._dx**2)

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

                p_0[xyz] = a_1[xyz] * p_1[xyz] + a_2[xyz] * p_2[xyz] + a_3[xyz] * self._lap(p_1, xyz)
                self._addSourceD2p(p_0, xyz, nt)
                self._readSensors(p_0, xyz, nt)

        @ti.kernel
        def circulate_buffers():
            for xyz in ti.grouped(p_0):
                # Circulate buffers
                p_2[xyz] = p_1[xyz]
                p_1[xyz] = p_0[xyz]

        t_init = time()
        for nt in range(self._n_steps):
            update_p(nt)
            circulate_buffers()
            if self._show_anim:
                self._show_anim_func(nt, p_0)
        sim_time = time() - t_init

        # "vx": vx, "vy": vy, "sens_vx": sens_vx, "sens_vy": sens_vy
        return {"sim_time": sim_time, "gpu_str": str(ti.lang.impl.current_cfg().arch),
                "sens_pressure": self._receiver.to_numpy(), "pressure": p_0.to_numpy()}

        

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
    # pass

except KeyError as key:
    print(f"Chave {key} nao encontrada no arquivo de configuracao.")

except ValueError as value:
    print(value)