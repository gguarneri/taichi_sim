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
# import matplotlib as mpl

# mpl.use('qt5agg')

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
        titype = ti.f32
        Nxyz = self._nx, self._ny
        Nc = self._deriv_acc
        Nt = self._n_steps

        self.c2 = ti.field(titype, shape=Nxyz)
        self.u_0 = ti.field(titype, shape=Nxyz)
        self.u_1 = ti.field(titype, shape=Nxyz)
        self.u_2 = ti.field(titype, shape=Nxyz)
        self.fd = ti.field(titype, shape=Nc)
        self.source = ti.field(titype, shape=Nt)
        self.pos_source = ti.field(titype, shape=Nxyz)

        self.fdcs = findiff.coefficients(deriv=2, acc=self._deriv_acc)['center']['coefficients'][self._deriv_acc // 2:]

        self.fd.from_numpy(self.fdcs.astype(np.float32))
        self.source.from_numpy(self._source_term.astype(np.float32))
        self.pos_source.from_numpy(self._pos_sources.astype(np.float32))

        # c2.fill(.4 ** 2)

        # for x in range(Nxyz[0]):
        #     for z in range(4):
        #         c2[x, z] = 0.
        #         c2[x, Nxyz[1]-1-z] = 0.
        # for z in range(Nxyz[1]):
        #     for x in range(4):
        #         c2[x, z] = 0.
        #         c2[Nxyz[0]-1-x, z] = 0.
        self.u_0.fill(0.)
        self.u_1.fill(0.)
        self.u_2.fill(0.)

        @ti.kernel
        def c2fill():
            for x, y in self.c2:
                if (
                    x < Nc - 1 or x > Nxyz[0] - Nc
                    or y < Nc - 1 or y > Nxyz[1] - Nc
                    ):
                    self.c2[x, y] = 0.
                else:
                    self.c2[x, y] = self._cp ** 2

        c2fill()

        # @ti.func
        # def mse(a, b):
        #     return ti.sum((a - b) ** 2)

        # @ti.func
        # def source(t):
        #     return 1e2*ti.exp(-(t-100)**2/500)*ti.cos(ti.math.pi/30*t)

        @ti.func
        def lap(x, y):
            a = 2 * self.fd[0] * self.u_1[x, y]
            for nc in range(1, Nc):
                a += self.fd[nc] * (self.u_1[x - nc, y] + self.u_1[x + nc, y] +
                                    self.u_1[x, y - nc] + self.u_1[x, y + nc])
            return a

        @ti.kernel
        def update_fields(nt: int):
            for xyz in ti.grouped(self.u_0):
                self.u_0[xyz] = 2 * self.u_1[xyz] - self.u_2[xyz] + self.c2[xyz] * lap(*xyz)
                #if (xyz - self.xyz_s).norm_sqr() < self.Dd2:
                #    self.u_0[xyz] += self.source(nt)
                if self.pos_source[xyz] == 0:
                    self.u_0[xyz] += self.source[nt]

        @ti.kernel
        def circulate_buffer():
            for xyz in ti.grouped(self.u_0):
                self.u_2[xyz] = self.u_1[xyz]
                self.u_1[xyz] = self.u_0[xyz]

        for nt in range(Nt):
            update_fields(nt)
            circulate_buffer()

        #return {"vx": vx, "vy": vy, "pressure": pressure,
        #        "sens_vx": sens_vx, "sens_vy": sens_vy, "sens_pressure": sens_pressure,
        #        "gpu_str": self._device.adapter.info["device"], "sim_time": sim_time}
        return {"sim_time": 0., "gpu_str": str(ti.lang.impl.current_cfg().arch),
                "sens_pressure": self.u_0.to_numpy()}

# ----------------------------------------------------------
# Avaliacao dos parametros na linha de comando
# ----------------------------------------------------------
parser = argparse.ArgumentParser()
# parser.add_argument('-c', '--config', help='Configuration file', default='config.json')
parser.add_argument('-c', '--config', help='Configuration file', default='/ensaios/ponto/ponto.json')
args = parser.parse_args()

# Cria a instancia do simulador
sim_instance = SimulatorTaichi(args.config)

#%% Executa simulacao
try:
    sim_instance.run()
    # pass

except KeyError as key:
    print(f"Chave {key} nao encontrada no arquivo de configuracao.")

except ValueError as value:
    print(value)