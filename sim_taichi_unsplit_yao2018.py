# =======================
# Importacao de pacotes de uso geral
# =======================
import argparse
from time import time
import numpy as np
# from sim_support import *
# from sim_support.simulator import Simulator

# ======================
# Importacao de pacotes especificos para a implementacao do simulador
# ======================
import taichi as ti
from sim_taichi_common import SimulatorTaichiCommon

# -----------------------------------------------------------------------------
# Aqui deve ser implementado o simulador como uma classe herdada de Simulator
# -----------------------------------------------------------------------------

@ti.data_oriented
class SimulatorTaichiUnsplit(SimulatorTaichiCommon):
    def __init__(self, file_config):
        # Chama do construtor padrao, que le o arquivo de configuracao
        super().__init__(file_config)

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

        D = ti.field(float, shape=self._Nxyz.to_numpy())
        tmp = np.zeros(self._Nxyz.to_numpy())
        tmp[1:-1, :] += self._a_x @ np.ones((1, self._Nxyz.to_numpy()[1]))
        tmp[:, 1:-1] += np.ones((self._Nxyz.to_numpy()[0], 1)) @ self._a_y

        D.from_numpy(tmp.astype(self._npFtp))

        p_0.fill(0.)
        p_1.fill(0.)
        p_2.fill(0.)

        dt2Odx2 = self._dt**2 / self._dx**2

        @ti.kernel
        def update_p(nt: int):
            for xyz in ti.grouped(p_0):
                d = D[xyz]
                oneMd = 1 - d
                a = (2 - d**2) / oneMd
                b = - (1 + d) / oneMd
                c =  dt2Odx2 / oneMd

                p_0[xyz] = a * p_1[xyz] + b * p_2[xyz] + c * self._c2[xyz] * self._lap(p_1, xyz)

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