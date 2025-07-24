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
        self._name = "Taichi Unsplit"
        
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
        # Convolutional Perfect Matched Layer (C-PML)
        # Auxiliary variables
        dp = []
        psi_p = []
        psi_dp = []
        for nd in range(self._Nd):
            Npml = self._Nxyz.to_numpy()
            Npml[nd] = self._Nabc[nd, 0] + self._Nabc[nd, 1]
            psi_p.append(ti.field(float, Npml))
            psi_dp.append(ti.field(float, Npml))

        dp = [ti.field(float, shape=self._Nxyz.to_numpy()) for _ in range(self._Nd)]
        # psi_dp = [ti.field(float, shape=self._Nxyz.to_numpy()) for _ in range(self._Nd)]
        # psi_p = [ti.field(float, shape=self._Nxyz.to_numpy()) for _ in range(self._Nd)]


        p_0.fill(0.)
        p_1.fill(0.)
        p_2.fill(0.)
        for nd in range(self._Nd):
            psi_dp[nd].fill(0.)
            psi_p[nd].fill(0.)
            dp[nd].fill(0.)

        dt2Odx2 = self._dt**2 / self._dx**2

        @ti.kernel
        def update_p(nt: int):
            for xyz in ti.grouped(p_0):
                tmp = 0.
                for nd in ti.static(range(self._Nd)):
                    D = self._D(dp[nd], xyz, nd, 1)
                    #psi_dp[nd][xyz] = self._b[nd][xyz] * psi_dp[nd][xyz] + (self._b[nd][xyz] - 1) * tmp2
                    tmp += psi_dp[nd][xyz] + self._pml(D, psi_dp[nd], xyz, nd)

                p_0[xyz] = 2 * p_1[xyz] - p_2[xyz] + dt2Odx2 * self._c2[xyz] * tmp

                self._addSourceD2p(p_0, xyz, nt)
                self._readSensors(p_0, xyz, nt)

                # Circulate buffers
                p_2[xyz] = p_1[xyz]
                p_1[xyz] = p_0[xyz]

        @ti.kernel
        def update_psis():
            for xyz in ti.grouped(p_0):
                for nd in ti.static(range(self._Nd)):
                    # dp[nd][xyz] = self._D(p_0, xyz, nd, 0)
                    D = self._D(p_0, xyz, nd, 0)
                    #psi_p[nd][xyz] = self._b[nd][xyz] * psi_p[nd][xyz] + (self._b[nd][xyz] - 1) * dp[nd][xyz]

                    dp[nd][xyz] += psi_p[nd][xyz]

        t_init = time()
        for nt in range(self._n_steps):
            update_psis()
            update_p(nt)
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