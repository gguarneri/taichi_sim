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
# from findiff import coefficients as fdcoeffs
# from sim_taichi_common import SimulatorTaichiCommon

# -----------------------------------------------------------------------------
# Aqui deve ser implementado o simulador como uma classe herdada de Simulator
# -----------------------------------------------------------------------------
class SimulatorTaichiStaggered(Simulator):
    def __init__(self, file_config):
        # Chama do construtor padrao, que le o arquivo de configuracao
        super().__init__(file_config, sim_model="split")

        # Define o nome do simulador
        self._name = "Taichi"

    def implementation(self):
        super().implementation()

        # --------------------------------------------
        # Aqui comeca o codigo especifico do simulador
        # --------------------------------------------

        ti.init(arch=ti.gpu, default_fp=ti.f32, default_ip=ti.i32)

        # Dimensions
        try: _Nxyz = (self._nx,); _Nxyz += (self._ny,); _Nxyz += (self._nz,)
        except AttributeError: pass
        Nd = len(_Nxyz)  # Number of dimensions
        Nxyz = ti.field(int, Nd)  # Dimensions in taichi field
        Nxyz.from_numpy(np.array(_Nxyz).astype(np.int32))

        # Pressure and velocity fields
        p = ti.field(float, _Nxyz)
        v = [ti.field(float, _Nxyz) for _ in range(Nd)]

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
            dp = self._source_term[ns] * self._dt * self._one_dx * self._one_dy
            source[ns].from_numpy(dp.astype(np.float32))
        receiver = ti.field(float, (self._n_steps, self._n_rec))

        # Absorbing Boundary Conditions (ABC)
        try:  # TODO: reavaliar ordem xyz, xzy, etc
            _Nabc = ((self._roi._pml_xmax_len, self._roi._pml_xmin_len),)
            _Nabc += ((self._roi._pml_zmax_len, self._roi._pml_zmin_len),)
            _Nabc += ((self._roi._pml_ymax_len, self._roi._pml_ymin_len),)
        except AttributeError: pass
        Nabc = ti.field(int, (3, 2))
        Nabc.from_numpy(np.array(_Nabc).astype(np.int32))

        # Convolutional Perfect Matched Layer (C-PML)
        # Auxiliary variables
        psi_p = []
        psi_v = []
        for nd in range(Nd):
            Npml = list(_Nxyz)
            Npml[nd] = _Nabc[nd][0] + _Nabc[nd][1]
            psi_p.append(ti.field(float, Npml))
            psi_v.append(ti.field(float, Npml))

        # C-PML coefficients (to be used with forward operator)
        bf = ti.field(float, np.max(_Nabc))
        bf.from_numpy(self._b_x[:_Nabc[0][0], 0])
        af = ti.field(float, np.max(_Nabc))
        af.from_numpy(self._a_x[:_Nabc[0][0], 0])
        kf = ti.field(float, np.max(_Nabc))
        kf.from_numpy(self._k_x[:_Nabc[0][0], 0])

        # C-PML coefficients (to be used with backward operator)
        bh = ti.field(float, np.max(_Nabc))
        bh.from_numpy(self._b_x_half[:_Nabc[0][0], 0])
        ah = ti.field(float, np.max(_Nabc))
        ah.from_numpy(self._a_x_half[:_Nabc[0][0], 0])
        kh = ti.field(float, np.max(_Nabc))
        kh.from_numpy(self._k_x_half[:_Nabc[0][0], 0])

        # Filling fields with zeros
        p.fill(0.)
        for nd in range(Nd):
            v[nd].fill(0.)
            psi_p[nd].fill(0.)
            psi_v[nd].fill(0.)

        # @ti.kernel
        # def zero_boundaries(prmtr: ti.template()):
        #     for xyz in ti.grouped(prmtr):
        #         cond = False
        #         for nd in ti.static(range(Nd)):
        #             cond = cond or xyz[nd] < self._deriv_acc - 1 or xyz[nd] > Nxyz[nd] - self._deriv_acc
        #         if cond:
        #             prmtr[xyz] = 0.

        # Bulk modulus in taichi field
        K = ti.field(float, _Nxyz)
        K.fill(self._cp**2 * self._rho * self._dt / self._dx)
        # zero_boundaries(K)

        # Inverse of density in taichi field
        rho_inv = ti.field(float, _Nxyz)
        rho_inv.fill(self._dt / (self._rho * self._dx))
        # zero_boundaries(rho_inv)

        @ti.func
        def D(u, xyz, nd: int, bf: int, imax):
            """
            Derivative operator

            Parameters
            ----------
            u: field
            xyz: coordinate
            nd: dimension
            bf: backward (0) or forward (1)
            imax: maximum index

            Returns
            -------
            d: derivative of field
            """
            d = 0.
            # imax = u.shape[nd[0]]
            for nc in ti.static(range(self._deriv_acc)):
                # # Solution 1
                # xyz_p = xyz[:]
                # xyz_n = xyz[:]
                # xyz_p[nd] += nc + bf
                # xyz_n[nd] -= nc - bf + 1
                # a = u[xyz_p] if xyz_p[nd] < imax else 0
                # b = u[xyz_n] if xyz_n[nd] >= 0 else 0
                # d += ti.static(self._coefs[nc]) * (a - b)

                # # Solution 2
                # xyz_tmp = xyz[:]
                # xyz_tmp[nd] += nc + bf
                # a = u[xyz_tmp] if xyz_tmp[nd] < imax else 0
                # xyz_tmp[nd] += - 2 * nc - 1
                # b = u[xyz_tmp] if xyz_tmp[nd] >= 0 else 0
                # d += ti.static(self._coefs[nc]) * (a - b)

                # c = u.shape[0]
                # ti.static_print(nd)
                # c = c[ti.static(nd)]

                # Solution 3
                xyz[nd] += nc + bf
                a = u[xyz] if xyz[nd] < imax else 0
                xyz[nd] += - 2 * nc - 1
                b = u[xyz] if xyz[nd] >= 0 else 0
                xyz[nd] += nc + 1 - bf
                d += ti.static(self._coefs[nc]) * (a - b)

            return d

        @ti.func
        def cpml(d, psi, xyz, nd: int, al, bl, kl, ar, br, kr):
            """
            CPML operator

            Parameters
            ----------
            d: derivative of field
            psi: auxiliary field
            xyz: coordinate
            nd: dimension
            al, bl, kl: left parameters
            ar, br, kr: right parameters

            Returns
            -------
            r: update term
            """
            r = d
            if xyz[nd] < Nabc[nd, 0]:
                i = xyz[nd]
                r = r / kl[i] + psi[xyz]
                psi[xyz] = bl[i] * psi[xyz] + al[i] * d
            elif xyz[nd] > Nxyz[nd] - Nabc[nd, 1] - 1:
                xyz_r = xyz[:]
                xyz_r[nd] += Nabc[nd, 1] - Nxyz[nd] + Nabc[nd, 0]
                i = Nxyz[nd] - xyz[nd] - 1
                r = r / kr[i] + psi[xyz_r]
                psi[xyz_r] = br[i] * psi[xyz_r] + ar[i] * d
            return r

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
        
        @ti.kernel
        def update_p(nt: int):
            for xyz in ti.grouped(p):
                for nd in ti.static(range(Nd)):
                    d = D(v[nd], xyz, nd, 1, v[nd].shape[nd])
                    p[xyz] -= K[xyz] * cpml(d, psi_v[nd], xyz, nd, ah, bh, kh, af, bf, kf)

                add_source(p, xyz, nt)
                read_sensors(p, xyz, nt)

        @ti.kernel
        def update_v():
            for xyz in ti.grouped(p):
                for nd in ti.static(range(Nd)):
                    d = D(p, xyz, nd, 0, p.shape[nd])
                    v[nd][xyz] -= rho_inv[xyz] * cpml(d, psi_p[nd], xyz, nd, af, bf, kf, ah, bh, kh)
                    # Dirichlet
                    if xyz[nd] < self._deriv_acc - 1 or xyz[nd] > Nxyz[nd] - self._deriv_acc:
                        v[nd][xyz] = 0.

        # Definicao dos limites para a plotagem dos campos
        v_max = 10.
        v_min = - v_max
        def show_anim_func(nt: int, u):
            if not nt % self._it_display:
                # TODO: reavaliar xyz
                u_np = u.to_numpy()[
                    self._roi.get_ix_min():self._roi.get_ix_max(), self._roi.get_iz_min():self._roi.get_iz_max()]
                self._windows_gpu[0].imv.setImage(u_np, levels=[v_min, v_max])
                self._app.processEvents()

        t_init = time()
        for nt in range(self._n_steps):
            update_p(nt)
            update_v()
            if self._show_anim:
                show_anim_func(nt, p)
        sim_time = time() - t_init

        # "vx": vx, "vy": vy, "sens_vx": sens_vx, "sens_vy": sens_vy
        return {"sim_time": sim_time, "gpu_str": str(ti.lang.impl.current_cfg().arch),
                "sens_pressure": receiver.to_numpy(), "pressure": p.to_numpy()}

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