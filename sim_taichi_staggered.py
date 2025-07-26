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
class SimulatorTaichiStaggered(SimulatorTaichiCommon):
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

        # Pressure and velocity fields
        p = ti.field(float, self._Nxyz.to_numpy())
        v = [ti.field(float, self._Nxyz.to_numpy()) for _ in range(self._Nd)]

        # Absorbing Boundary Conditions (ABC)
        # Convolutional Perfect Matched Layer (C-PML)
        # Auxiliary variables
        psi_p = []
        psi_v = []
        for nd in range(self._Nd):
            Npml = self._Nxyz.to_numpy()
            Npml[nd] = self._Nabc[nd, 0] + self._Nabc[nd, 1]
            psi_p.append(ti.field(float, Npml))
            psi_v.append(ti.field(float, Npml))

        # Filling fields with zeros
        p.fill(0.)
        for nd in range(self._Nd):
            v[nd].fill(0.)
            psi_p[nd].fill(0.)
            psi_v[nd].fill(0.)

        dtOdx = self._dt / self._dx
        
        @ti.kernel
        def update_p(nt: int):
            for xyz in ti.grouped(p):
                for nd in ti.static(range(self._Nd)):
                    D = self._D(v[nd], xyz, nd, 1, v[nd].shape[nd])
                    p[xyz] -= dtOdx * self._c2[xyz]  * self._pml(D, psi_v[nd], xyz, nd)

                self._addSourceDp(p, xyz, nt)
                self._readSensors(p, xyz, nt)

        @ti.kernel
        def update_v():
            for xyz in ti.grouped(p):
                for nd in ti.static(range(self._Nd)):
                    D = self._D(p, xyz, nd,0, p.shape[nd])
                    v[nd][xyz] -= dtOdx * self._pml(D, psi_p[nd], xyz, nd)

        t_init = time()
        for nt in range(self._n_steps):
            update_v()
            update_p(nt)
            if self._show_anim:
                self._show_anim_func(nt, p)
        sim_time = time() - t_init

        # "vx": vx, "vy": vy, "sens_vx": sens_vx, "sens_vy": sens_vy
        return {"sim_time": sim_time, "gpu_str": str(ti.lang.impl.current_cfg().arch),
                "sens_pressure": self._receiver.to_numpy(), "pressure": p.to_numpy()}

# ----------------------------------------------------------
# Avaliacao dos parametros na linha de comando
# ----------------------------------------------------------
parser = argparse.ArgumentParser()
# parser.add_argument('-c', '--config', help='Configuration file', default='config.json')
default_config_file = "ensaios/ponto/ponto.json"
parser.add_argument('-c', '--config', help='Configuration file', default=default_config_file)

args = parser.parse_args()

# Cria a instancia do simulador
sim_instance = SimulatorTaichiStaggered(args.config)

#%% Executa simulacao
try:
    sim_instance.run()
    # pass

except KeyError as key:
    print(f"Chave {key} nao encontrada no arquivo de configuracao.")

except ValueError as value:
    print(value)