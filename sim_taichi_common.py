# from sim_support.simulator import Simulator
import numpy as np
import taichi as ti
from findiff import coefficients as fdcoeffs

# view = 10
# vmm = 1e6
# cmap = mpl.colormaps['bwr']
# norm = lambda x: .5 + x / (2 * vmm)
# window = ti.ui.Window(name="Wave", res=st.Nxyz[0:2], pos=(0, 0))
# canvas = window.get_canvas()
# Definicao dos limites para a plotagem dos campos

deriv_acc = None
fdstg = None
Ncstg = 0
Nxyz = None
Nd = None
xyz_s = None
xyz_r = None
tiFtype = ti.float32
npFtype = np.float32
Ns = 0
Nr = 0
b = None
v_max = None
v_min = None
ix_min = None
ix_max = None
iy_min = None
iy_max = None

def init(self):
    global deriv_acc, fdstg, Ncstg, Nxyz, Nd, xyz_s, xyz_r, Ns, Nr, b
    global v_max, v_min, ix_min, ix_max, iy_min, iy_max

    deriv_acc = self._deriv_acc
    offsets = tuple(np.arange(-self._deriv_acc, self._deriv_acc) + .5)
    fd_np = fdcoeffs(deriv=1, offsets=offsets)["coefficients"][self._deriv_acc:]
    fdstg = tuple(fd_np.astype(npFtype))
    Ncstg = len(fdstg)

    try:
        Nxyz = self._nx, self._ny, self._nz
        Nd = len(Nxyz)  # Number of dimensions
        xyz_s = self._ix_src, self._iy_src, self._iz_src
        xyz_r = self._ix_rec, self._iy_rec, self._iz_rec
        # Nxyz_abc = (((self._roi._pml_xmin_len, self._ny, self._nz), (self._roi._pml_xmax_len, self._ny, self._nz)),
        #             ((self._nx, self._roi._pml_ymin_len, self._nz), (self._nx, self._roi._pml_ymax_len, self._nz)),
        #             ((self._nx, self._ny, self._roi._pml_ymin_len), (self._nx, self._ny, self._roi._pml_ymax_len)))

    except AttributeError:
        Nxyz = self._nx, self._ny
        Nd = len(Nxyz)  # Number of dimensions
        xyz_s = self._ix_src, self._iy_src
        xyz_r = self._ix_rec, self._iy_rec
        # Nxyz_abc = (((self._roi._pml_xmin_len, self._ny), (self._roi._pml_xmax_len, self._ny)),
        #             ((self._nx, self._roi._pml_ymin_len), (self._nx, self._roi._pml_ymax_len)))
        bx = np.zeros((Nxyz[0], 1), dtype=npFtype)
        by = np.zeros((1, Nxyz[1]), dtype=npFtype)
        bx[1:-1, :] = self._b_x
        by[:, 1:-1] = self._b_y
        b = [None, None]
        b[0] = (bx @ np.ones(Nxyz[1])[np.newaxis]).astype(npFtype)
        b[1] = (np.ones(Nxyz[0])[:, np.newaxis] @ by).astype(npFtype)

    xyz_s = tuple(tuple(i) for i in np.array(xyz_s).T)  # Coordinates of sources
    xyz_r = tuple(tuple(i) for i in np.array(xyz_r).T)  # Coordinates of receivers

    Ns = len(xyz_s)  # Number of sources
    Nr = len(xyz_r)  # Number of receivers


    # Definicao dos limites para a plotagem dos campos
    v_max = 1.0
    v_min = - v_max
    ix_min = self._roi.get_ix_min()
    ix_max = self._roi.get_ix_max()
    iy_min = self._roi.get_iz_min()
    iy_max = self._roi.get_iz_max()

def show_anim(self, nt, u):
    if self._show_anim:
        if not nt % self._it_display:
            u_np = u.to_numpy()[ix_min:ix_max, iy_min:iy_max]
            self._windows_gpu[0].imv.setImage(u_np, levels=[v_min, v_max])
            self._app.processEvents()

@ti.func
def D(nd, u, xyz, bf):
    d = 0.
    for nc in ti.static(range(Ncstg)):
        # # Solution 1
        # xyz_p = xyz[:]
        # xyz_m = xyz[:]
        # xyz_p[nd] += nc + bf
        # xyz_m[nd] -= nc - bf + 1
        # d += ti.static(fdstg[nc]) * (u[xyz_p] - u[xyz_m])

        # # Solution 2
        # xyz[nd] += nc + bf
        # a = u[xyz]
        # xyz[nd] += - 2 * nc - 1
        # d += ti.static(fdstg[nc]) * (a - u[xyz])
        # xyz[nd] += nc + 1 - bf

        # Solution 3
        xyz_tmp = xyz[:]
        xyz_tmp[nd] += nc + bf
        a = u[xyz_tmp]
        xyz_tmp[nd] += - 2 * nc - 1
        d += ti.static(fdstg[nc]) * (a - u[xyz_tmp])
    return d


@ti.kernel
def parameters_zero_boundaries(prmtr: ti.template()):
    for xyz in ti.grouped(prmtr):
        cond = False
        for nd in ti.static(range(Nd)):
            cond = cond or xyz[nd] < deriv_acc or xyz[nd] >= Nxyz[nd] - deriv_acc
        if cond:
            prmtr[xyz] = 0.

# @ti.func
# def lap(u, xyz):
#     lp = ti.static(Nd) * ti.static(fd[0]) * u[xyz]
#     for nc in ti.static(range(1, Nc)):  # compile-time loop unrolling
#         for nd in ti.static(range(Nd)):
#             xyz[nd] += nc
#             a = u[xyz]
#             xyz[nd] -= 2 * nc
#             lp += ti.static(fd[nc]) * (a + u[xyz])
#             xyz[nd] += nc
#     return lp
