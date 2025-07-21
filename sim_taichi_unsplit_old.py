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
        import sim_taichi_common_old as st

        st.init(self)
        ti.init(arch=ti.gpu)

        c2 = ti.field(st.tiFtype, shape=st.Nxyz)
        p_0 = ti.field(st.tiFtype, shape=st.Nxyz)
        p_1 = ti.field(st.tiFtype, shape=st.Nxyz)
        p_2 = ti.field(st.tiFtype, shape=st.Nxyz)
        phi_dp = [ti.field(st.tiFtype, shape=st.Nxyz) for _ in range(st.Nd)]
        phi_p = [ti.field(st.tiFtype, shape=st.Nxyz) for _ in range(st.Nd)]
        dp = [ti.field(st.tiFtype, shape=st.Nxyz) for _ in range(st.Nd)]
        b = [ti.field(st.tiFtype, shape=st.Nxyz) for _ in range(st.Nd)]
        source = ti.field(st.tiFtype, shape=self._n_steps)
        source.from_numpy((self._dt**2 * self._source_term).astype(st.npFtype))
        receiver = ti.field(st.tiFtype, shape=(self._n_steps, self._n_rec))

        p_0.fill(0.)
        p_1.fill(0.)
        p_2.fill(0.)
        for nd in range(st.Nd):
            phi_dp[nd].fill(0.)
            phi_p[nd].fill(0.)
            dp[nd].fill(0.)
            b[nd].from_numpy(st.b[nd])

        c2.fill(self._cp**2 * self._dt**2 / self._dx**2)
        st.parameters_zero_boundaries(c2)

        @ti.kernel
        def update_p(nt: int):
            for xyz in ti.grouped(p_0):
                tmp1 = 0.
                for nd in ti.static(range(st.Nd)):
                    tmp2 = st.D(nd, dp[nd], xyz, 1)
                    phi_dp[nd][xyz] = b[nd][xyz] * phi_dp[nd][xyz] + (b[nd][xyz] - 1) * tmp2
                    tmp1 += phi_dp[nd][xyz] + tmp2

                p_0[xyz] = 2 * p_1[xyz] - p_2[xyz] + c2[xyz] * tmp1

                for ns in ti.static(range(st.Ns)):
                    if all(xyz == st.xyz_s[ns]):
                        p_0[xyz] += source[nt]
                for nr in ti.static(range(st.Nr)):
                    if all(xyz == st.xyz_r[nr]):
                        receiver[nt, nr] = p_0[xyz]

                # Circulate buffers
                p_2[xyz] = p_1[xyz]
                p_1[xyz] = p_0[xyz]

        @ti.kernel
        def update_phi():
            for xyz in ti.grouped(p_0):
                for nd in ti.static(range(st.Nd)):
                    dp[nd][xyz] = st.D(nd, p_0, xyz, 0)
                    phi_p[nd][xyz] = b[nd][xyz] * phi_p[nd][xyz] + (b[nd][xyz] - 1) * dp[nd][xyz]
                    dp[nd][xyz] += phi_p[nd][xyz]

        t_init = time()
        for nt in range(self._n_steps):
            update_phi()
            update_p(nt)
            st.show_anim(self, nt, p_0)
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