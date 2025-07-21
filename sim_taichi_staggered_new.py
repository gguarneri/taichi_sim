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
import sim_taichi_common_new as stic

# -----------------------------------------------------------------------------
# Aqui deve ser implementado o simulador como uma classe herdada de Simulator
# -----------------------------------------------------------------------------
class SimulatorTaichiStaggered(stic.SimulatorTaichiCommon):
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

        p = ti.field(self._tiFtype, shape=self._Nxyz)
        v = [ti.field(self._tiFtype, shape=self._Nxyz) for _ in range(len(self._Nxyz))]
        psi_p = [ti.field(self._tiFtype, shape=self._Nxyz) for _ in range(len(self._Nxyz))]
        psi_v = [ti.field(self._tiFtype, shape=self._Nxyz) for _ in range(len(self._Nxyz))]

        p.fill(0.)
        for nd in range(len(self._Nxyz)):
            v[nd].fill(0.)
            psi_p[nd].fill(0.)
            psi_v[nd].fill(0.)
        
        dtOdx = self._dt / self._dx
        
        @ti.kernel
        def update_p(nt: int):
            for xyz in ti.grouped(p):
                for nd in ti.static(range(len(self._Nxyz))):
                    tmp = self._D(nd, v[nd], xyz, 1)
                    p[xyz] -= dtOdx * self._c2[xyz] * (tmp + psi_v[nd][xyz])
                    psi_v[nd][xyz] = self._b[nd][xyz] * psi_v[nd][xyz] + (self._b[nd][xyz] - 1) * tmp
                for ns in ti.static(range(self._n_src)):
                    if all(xyz == self._xyz_s[ns]):
                        p[xyz] += self._source_dp[ns][nt]
                for nr in ti.static(range(self._n_src)):
                    if all(xyz == self._xyz_r[nr]):
                        self._receiver[nt, nr] = p[xyz]

        @ti.kernel
        def update_v():
            for xyz in ti.grouped(p):
                for nd in ti.static(range(len(self._Nxyz))):
                    tmp = self._D(nd, p, xyz, 0)
                    v[nd][xyz] -= dtOdx * (tmp + psi_p[nd][xyz])
                    psi_p[nd][xyz] = self._b[nd][xyz] * psi_p[nd][xyz] + (self._b[nd][xyz] - 1) * tmp

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

except KeyError as key:
    print(f"Chave {key} nao encontrada no arquivo de configuracao.")

except ValueError as value:
    print(value)