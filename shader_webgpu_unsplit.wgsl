// ++++++++++++++++++++++++++++++
// +++ Constants to override ++++
// ++++++++++++++++++++++++++++++
// Workgroup size for x dimension
const wsx : u32 = _WSX_;

// Workgroup size for y dimension
const wsy : u32 = _WSY_;

// Workgroup size for sensors store kernel
const idx_rec_offset: u32 = _IDX_REC_OFFSET_;

// +++++++++++++++++++++++++++++++
// +++ Index access functions ++++
// +++++++++++++++++++++++++++++++
// function to convert 2D [i,j] index into 1D [] index
fn ij(i: i32, j: i32, i_max: i32, j_max: i32) -> i32 {
    let index = j + i * j_max;

    return select(-1, index, i >= 0 && i < i_max && j >= 0 && j < j_max);
}

// ++++++++++++++++++++++++++++++
// ++++ Group 0 - parameters ++++
// ++++++++++++++++++++++++++++++
struct SimIntValues {
    x_sz: i32,          // x field size
    y_sz: i32,          // y field size
    n_iter: i32,        // num iterations
    n_src_el: i32,      // num probes tx elements
    n_rec_el: i32,      // num probes rx elements
    n_rec_pt: i32,      // num rec pto
    fd_coeff: i32,      // num fd coefficients
    it: i32             // time iteraction
};

@group(0) @binding(0) // param_int32
var<storage,read_write> sim_int_par: SimIntValues;

// ----------------------------------

struct SimFltValues {
    dx: f32,            // delta x
    dy: f32,            // delta y
    dt: f32,            // delta t
};

@group(0) @binding(1)   // param_flt32
var<storage,read> sim_flt_par: SimFltValues;

// -----------------------------------
// --- Force array access funtions ---
// -----------------------------------
@group(0) @binding(2) // source term
var<storage,read> source_term: array<f32>;

// function to get a source_term array value
fn get_source_term(n: i32, e: i32) -> f32 {
    let index: i32 = ij(n, e, sim_int_par.n_iter, sim_int_par.n_src_el);

    return select(0.0, source_term[index], index != -1);
}

// ----------------------------------

@group(0) @binding(3) // source term index
var<storage,read> idx_src: array<i32>;

// function to get a source term index of a source
fn get_idx_source_term(x: i32, y: i32) -> i32 {
    let index: i32 = ij(x, y, sim_int_par.x_sz, sim_int_par.y_sz);

    return select(-1, idx_src[index], index != -1);
}

// -------------------------------------------------
// --- CPML X coefficients array access funtions ---
// -------------------------------------------------
@group(0) @binding(4) // a_x
var<storage,read> a_x: array<f32>;

// function to get a a_x array value
fn get_a_x(n: i32) -> f32 {
    let pad: i32 = (sim_int_par.fd_coeff - 1) * 2;

    return select(0.0, a_x[n], n >= 0 && n < sim_int_par.x_sz - pad);
}

// ----------------------------------

@group(0) @binding(5) // b_x
var<storage,read> b_x: array<f32>;

// function to get a b_x array value
fn get_b_x(n: i32) -> f32 {
    let pad: i32 = (sim_int_par.fd_coeff - 1) * 2;

    return select(0.0, b_x[n], n >= 0 && n < sim_int_par.x_sz - pad);
}

// ----------------------------------

@group(0) @binding(6) // k_x
var<storage,read> k_x: array<f32>;

// function to get a k_x array value
fn get_k_x(n: i32) -> f32 {
    let pad: i32 = (sim_int_par.fd_coeff - 1) * 2;

    return select(0.0, k_x[n], n >= 0 && n < sim_int_par.x_sz - pad);
}

// ----------------------------------

@group(0) @binding(7) // a_x_h
var<storage,read> a_x_h: array<f32>;

// function to get a a_x_h array value
fn get_a_x_h(n: i32) -> f32 {
    let pad: i32 = (sim_int_par.fd_coeff - 1) * 2;

    return select(0.0, a_x_h[n], n >= 0 && n < sim_int_par.x_sz - pad);
}

// ----------------------------------

@group(0) @binding(8) // b_x_h
var<storage,read> b_x_h: array<f32>;

// function to get a b_x_h array value
fn get_b_x_h(n: i32) -> f32 {
    let pad: i32 = (sim_int_par.fd_coeff - 1) * 2;

    return select(0.0, b_x_h[n], n >= 0 && n < sim_int_par.x_sz - pad);
}

// ----------------------------------

@group(0) @binding(9) // k_x_h
var<storage,read> k_x_h: array<f32>;

// function to get a k_x_h array value
fn get_k_x_h(n: i32) -> f32 {
    let pad: i32 = (sim_int_par.fd_coeff - 1) * 2;

    return select(0.0, k_x_h[n], n >= 0 && n < sim_int_par.x_sz - pad);
}

// -------------------------------------------------
// --- CPML Y coefficients array access funtions ---
// -------------------------------------------------
@group(0) @binding(10) // a_y
var<storage,read> a_y: array<f32>;

// function to get a a_y array value
fn get_a_y(n: i32) -> f32 {
    let pad: i32 = (sim_int_par.fd_coeff - 1) * 2;

    return select(0.0, a_y[n], n >= 0 && n < sim_int_par.y_sz - pad);
}

// ----------------------------------

@group(0) @binding(11) // b_y
var<storage,read> b_y: array<f32>;

// function to get a b_y array value
fn get_b_y(n: i32) -> f32 {
    let pad: i32 = (sim_int_par.fd_coeff - 1) * 2;

    return select(0.0, b_y[n], n >= 0 && n < sim_int_par.y_sz - pad);
}

// ----------------------------------

@group(0) @binding(12) // k_y
var<storage,read> k_y: array<f32>;

// function to get a k_y array value
fn get_k_y(n: i32) -> f32 {
    let pad: i32 = (sim_int_par.fd_coeff - 1) * 2;

    return select(0.0, k_y[n], n >= 0 && n < sim_int_par.y_sz - pad);
}

// ----------------------------------

@group(0) @binding(13) // a_y_h
var<storage,read> a_y_h: array<f32>;

// function to get a a_y_h array value
fn get_a_y_h(n: i32) -> f32 {
    let pad: i32 = (sim_int_par.fd_coeff - 1) * 2;

    return select(0.0, a_y_h[n], n >= 0 && n < sim_int_par.y_sz - pad);
}

// ----------------------------------

@group(0) @binding(14) // b_y_h
var<storage,read> b_y_h: array<f32>;

// function to get a b_y_h array value
fn get_b_y_h(n: i32) -> f32 {
    let pad: i32 = (sim_int_par.fd_coeff - 1) * 2;

    return select(0.0, b_y_h[n], n >= 0 && n < sim_int_par.y_sz - pad);
}

// ----------------------------------

@group(0) @binding(15) // k_y_h
var<storage,read> k_y_h: array<f32>;

// function to get a k_y_h array value
fn get_k_y_h(n: i32) -> f32 {
    let pad: i32 = (sim_int_par.fd_coeff - 1) * 2;

    return select(0.0, k_y_h[n], n >= 0 && n < sim_int_par.y_sz - pad);
}

// -------------------------------------------------------------
// --- Finite difference index limits arrays access funtions ---
// -------------------------------------------------------------
@group(0) @binding(16) // idx_fd
var<storage,read> idx_fd: array<i32>;

// function to get an index to ini-half grid
fn get_idx_ih(c: i32) -> i32 {
    let index: i32 = ij(c, 0, sim_int_par.fd_coeff, 4);

    return select(-1, idx_fd[index], index != -1);
}

// function to get an index to ini-full grid
fn get_idx_if(c: i32) -> i32 {
    let index: i32 = ij(c, 1, sim_int_par.fd_coeff, 4);

    return select(-1, idx_fd[index], index != -1);
}

// function to get an index to fin-half grid
fn get_idx_fh(c: i32) -> i32 {
    let index: i32 = ij(c, 2, sim_int_par.fd_coeff, 4);

    return select(-1, idx_fd[index], index != -1);
}

// function to get an index to fin-full grid
fn get_idx_ff(c: i32) -> i32 {
    let index: i32 = ij(c, 3, sim_int_par.fd_coeff, 4);

    return select(-1, idx_fd[index], index != -1);
}

// ----------------------------------

@group(0) @binding(17) // fd_coeff
var<storage,read> fd_coeffs: array<f32>;

// function to get a fd coefficient
fn get_fdc(c: i32) -> f32 {
    return select(0.0, fd_coeffs[c], c >= 0 && c < (sim_int_par.fd_coeff * 2));
}

// ---------------------------------
// --- Rho X map access funtions ---
// ---------------------------------
@group(0) @binding(18) // rho_x
var<storage,read> rho_x_map: array<f32>;

// function to get a rho_x value
fn get_rho_x(x: i32, y: i32) -> f32 {
    let index: i32 = ij(x, y, sim_int_par.x_sz, sim_int_par.y_sz);

    return select(0.0, rho_x_map[index], index != -1);
}

// ---------------------------------
// --- Rho Y map access funtions ---
// ---------------------------------
@group(0) @binding(19) // rho_y
var<storage,read> rho_y_map: array<f32>;

// function to get a rho_y value
fn get_rho_y(x: i32, y: i32) -> f32 {
    let index: i32 = ij(x, y, sim_int_par.x_sz, sim_int_par.y_sz);

    return select(0.0, rho_y_map[index], index != -1);
}

// ---------------------------------
// --- Kappa map access funtions ---
// ---------------------------------
@group(0) @binding(20) // kappa
var<storage,read> kappa_map: array<f32>;

// function to get a kappa value
fn get_kappa(x: i32, y: i32) -> f32 {
    let index: i32 = ij(x, y, sim_int_par.x_sz, sim_int_par.y_sz);

    return select(0.0, kappa_map[index], index != -1);
}

// +++++++++++++++++++++++++++++++++++++
// ++++ Group 1 - simulation arrays ++++
// +++++++++++++++++++++++++++++++++++++
// ---------------------------------------
// --- Pressure fields arrays access funtions ---
// ---------------------------------------
@group(1) @binding(0) // past pressure field
var<storage,read_write> press_past: array<f32>;

// function to get a press_past array value
fn get_press_past(x: i32, y: i32) -> f32 {
    let index: i32 = ij(x, y, sim_int_par.x_sz, sim_int_par.y_sz);

    return select(0.0, press_past[index], index != -1);
}

// function to set a press_past array value
fn set_press_past(x: i32, y: i32, val : f32) {
    let index: i32 = ij(x, y, sim_int_par.x_sz, sim_int_par.y_sz);

    if(index != -1) {
        press_past[index] = val;
    }
}

// ----------------------------------

@group(1) @binding(1) // present pressure field
var<storage,read_write> press_pr: array<f32>;

// function to get a press_pr array value
fn get_press_pr(x: i32, y: i32) -> f32 {
    let index: i32 = ij(x, y, sim_int_par.x_sz, sim_int_par.y_sz);

    return select(0.0, press_pr[index], index != -1);
}

// function to set a press_pr array value
fn set_press_pr(x: i32, y: i32, val : f32) {
    let index: i32 = ij(x, y, sim_int_par.x_sz, sim_int_par.y_sz);

    if(index != -1) {
        press_pr[index] = val;
    }
}

// ----------------------------------

@group(1) @binding(2) // future pressure field
var<storage,read_write> press_ft: array<f32>;

// function to get a press_ft array value
fn get_press_ft(x: i32, y: i32) -> f32 {
    let index: i32 = ij(x, y, sim_int_par.x_sz, sim_int_par.y_sz);

    return select(0.0, press_ft[index], index != -1);
}

// function to set a press_ft array value
fn set_press_ft(x: i32, y: i32, val : f32) {
    let index: i32 = ij(x, y, sim_int_par.x_sz, sim_int_par.y_sz);

    if(index != -1) {
        press_ft[index] = val;
    }
}

// ----------------------------------

@group(1) @binding(3) // p_2
var<storage,read_write> p_2: f32;

// -------------------------------------
// --- Memory arrays access funtions ---
// -------------------------------------
@group(1) @binding(4) // mdpx_dx field
var<storage,read_write> mdpx_dx: array<f32>;

// function to get a mdpx_dx array value
fn get_mdpx_dx(x: i32, y: i32) -> f32 {
    let index: i32 = ij(x, y, sim_int_par.x_sz, sim_int_par.y_sz);

    return select(0.0, mdpx_dx[index], index != -1);
}

// function to set a mdpx_dx array value
fn set_mdpx_dx(x: i32, y: i32, val : f32) {
    let index: i32 = ij(x, y, sim_int_par.x_sz, sim_int_par.y_sz);

    if(index != -1) {
        mdpx_dx[index] = val;
    }
}

// ----------------------------------

@group(1) @binding(5) // mdpy_dy field
var<storage,read_write> mdpy_dy: array<f32>;

// function to get a mdpy_dy array value
fn get_mdpy_dy(x: i32, y: i32) -> f32 {
    let index: i32 = ij(x, y, sim_int_par.x_sz, sim_int_par.y_sz);

    return select(0.0, mdpy_dy[index], index != -1);
}

// function to set a mdpy_dy array value
fn set_mdpy_dy(x: i32, y: i32, val : f32) {
    let index: i32 = ij(x, y, sim_int_par.x_sz, sim_int_par.y_sz);

    if(index != -1) {
        mdpy_dy[index] = val;
    }
}

// ----------------------------------

@group(1) @binding(6) // mdpxx_dx field
var<storage,read_write> mdpxx_dx: array<f32>;

// function to get a mdpxx_dx array value
fn get_mdpxx_dx(x: i32, y: i32) -> f32 {
    let index: i32 = ij(x, y, sim_int_par.x_sz, sim_int_par.y_sz);

    return select(0.0, mdpxx_dx[index], index != -1);
}

// function to set a mdpxx_dx array value
fn set_mdpxx_dx(x: i32, y: i32, val : f32) {
    let index: i32 = ij(x, y, sim_int_par.x_sz, sim_int_par.y_sz);

    if(index != -1) {
        mdpxx_dx[index] = val;
    }
}

// ----------------------------------

@group(1) @binding(7) // mdpyy_dy field
var<storage,read_write> mdpyy_dy: array<f32>;

// function to get a mdpyy_dy array value
fn get_mdpyy_dy(x: i32, y: i32) -> f32 {
    let index: i32 = ij(x, y, sim_int_par.x_sz, sim_int_par.y_sz);

    return select(0.0, mdpyy_dy[index], index != -1);
}

// function to set a mdpyy_dy array value
fn set_mdpyy_dy(x: i32, y: i32, val : f32) {
    let index: i32 = ij(x, y, sim_int_par.x_sz, sim_int_par.y_sz);

    if(index != -1) {
        mdpyy_dy[index] = val;
    }
}

// ----------------------------------

@group(1) @binding(8) // dpx_dx field
var<storage,read_write> dpx_dx: array<f32>;

// function to get a dpx_dx array value
fn get_dpx_dx(x: i32, y: i32) -> f32 {
    let index: i32 = ij(x, y, sim_int_par.x_sz, sim_int_par.y_sz);

    return select(0.0, dpx_dx[index], index != -1);
}

// function to set a dpx_dx array value
fn set_dpx_dx(x: i32, y: i32, val : f32) {
    let index: i32 = ij(x, y, sim_int_par.x_sz, sim_int_par.y_sz);

    if(index != -1) {
        dpx_dx[index] = val;
    }
}

// ----------------------------------

@group(1) @binding(9) // dpy_dy field
var<storage,read_write> dpy_dy: array<f32>;

// function to get a dpy_dy array value
fn get_dpy_dy(x: i32, y: i32) -> f32 {
    let index: i32 = ij(x, y, sim_int_par.x_sz, sim_int_par.y_sz);

    return select(0.0, dpy_dy[index], index != -1);
}

// function to set a dpy_dy array value
fn set_dpy_dy(x: i32, y: i32, val : f32) {
    let index: i32 = ij(x, y, sim_int_par.x_sz, sim_int_par.y_sz);

    if(index != -1) {
        dpy_dy[index] = val;
    }
}

// +++++++++++++++++++++++++++++++++++++++++++++++
// ++++ Group 2 - sensors arrays and energies ++++
// +++++++++++++++++++++++++++++++++++++++++++++++
// --------------------------------------
// --- Sensors arrays access funtions ---
// --------------------------------------
@group(2) @binding(0) // sensors signals pressure
var<storage,read_write> sensors_pressure: array<f32>;

// function to get a sens_pressure array value
fn get_sens_pressure(n: i32, s: i32) -> f32 {
    let index: i32 = ij(n, s, sim_int_par.n_iter, sim_int_par.n_rec_el);

    return select(0.0, sensors_pressure[index], index != -1);
}

// function to set a sens_pressure array value
fn set_sens_pressure(n: i32, s: i32, val : f32) {
    let index: i32 = ij(n, s, sim_int_par.n_iter, sim_int_par.n_rec_el);

    if(index != -1) {
        sensors_pressure[index] = val;
    }
}

// ----------------------------------

@group(2) @binding(1) // delay sensor
var<storage,read> delay_rec: array<i32>;

// function to get a delay receiver value
fn get_delay_rec(s: i32) -> i32 {
    return select(0, delay_rec[s], s >= 0 && s < sim_int_par.n_rec_el);
}

// ----------------------------------

@group(2) @binding(2) // info rec ptos
var<storage,read> idx_sen: array<i32>;

// function to get an index of a sensor
fn get_idx_sensor(x: i32, y: i32) -> i32 {
    let index: i32 = ij(x, y, sim_int_par.x_sz, sim_int_par.y_sz);

    return select(-1, idx_sen[index], index != -1);
}

// ---------------
// --- Kernels ---
// ---------------
@compute
@workgroup_size(wsx, wsy)
fn teste_kernel(@builtin(global_invocation_id) index: vec3<u32>) {
    let x: i32 = i32(index.x);          // x thread index
    let y: i32 = i32(index.y);          // y thread index
    let last: i32 = sim_int_par.fd_coeff - 1;
    let offset_x: i32 = sim_int_par.fd_coeff - 1;
    let offset_y: i32 = sim_int_par.fd_coeff - 1;

    // Normal stresses
    var id_x_i: i32 = -get_idx_fh(last);
    var id_x_f: i32 = sim_int_par.x_sz - get_idx_ih(last);
    var id_y_i: i32 = -get_idx_ff(last);
    var id_y_f: i32 = sim_int_par.y_sz - get_idx_if(last);
    if(x >= id_x_i && x < id_x_f && y >= id_y_i && y < id_y_f) {
        set_press_ft(x, y, f32(wsx));
        set_press_pr(x, y, f32(wsy));
    }
}

// Kernel to calculate the 1st derivative of pressure
@compute
@workgroup_size(wsx, wsy)
fn pressure_first_der_kernel(@builtin(global_invocation_id) index: vec3<u32>) {
    let x: i32 = i32(index.x);          // x thread index
    let y: i32 = i32(index.y);          // y thread index
    let one_dx: f32 = 1.0 / sim_flt_par.dx;
    let one_dy: f32 = 1.0 / sim_flt_par.dy;
    let ord: i32 = sim_int_par.fd_coeff;
    let last: i32 = ord - 1;
    let offset: i32 = ord - 1;

    // Pressure
    p_2 = 0.0;
    var id_x_i: i32 = -get_idx_fh(last);
    var id_x_f: i32 = sim_int_par.x_sz - get_idx_ih(last);
    var id_y_i: i32 = -get_idx_fh(last);
    var id_y_f: i32 = sim_int_par.y_sz - get_idx_ih(last);
    if(x >= id_x_i && x < id_x_f && y >= id_y_i && y < id_y_f) {
        var vdpx_dx: f32 = 0.0;
        var vdpy_dy: f32 = 0.0;
        for(var c: i32 = 0; c < (ord * 2); c++) {
            let off: i32 = c - (ord - 1);
            vdpx_dx += get_fdc(c) * get_press_pr(x + off, y) * one_dx;
            vdpy_dy += get_fdc(c) * get_press_pr(x, y + off) * one_dy;
        }

        var mdpx_dx_new: f32 = get_b_x_h(x - offset) * get_mdpx_dx(x, y) + get_a_x_h(x - offset) * vdpx_dx;
        var mdpy_dy_new: f32 = get_b_y_h(y - offset) * get_mdpy_dy(x, y) + get_a_y_h(y - offset) * vdpy_dy;

        vdpx_dx = vdpx_dx/get_k_x_h(x - offset) + mdpx_dx_new;
        vdpy_dy = vdpy_dy/get_k_y_h(y - offset) + mdpy_dy_new;

        set_mdpx_dx(x, y, mdpx_dx_new);
        set_mdpy_dy(x, y, mdpy_dy_new);

        set_dpx_dx(x, y, vdpx_dx / get_rho_x(x, y));
        set_dpy_dy(x, y, vdpy_dy / get_rho_y(x, y));
    }
}

// Kernel to calculate the 2nd derivative of pressure
@compute
@workgroup_size(wsx, wsy)
fn pressure_second_der_kernel(@builtin(global_invocation_id) index: vec3<u32>) {
    let x: i32 = i32(index.x);          // x thread index
    let y: i32 = i32(index.y);          // y thread index
    let one_dx: f32 = 1.0 / sim_flt_par.dx;
    let one_dy: f32 = 1.0 / sim_flt_par.dy;
    let dt: f32 = sim_flt_par.dt;
    let ord: i32 = sim_int_par.fd_coeff;
    let last: i32 = ord - 1;
    let offset: i32 = ord - 1;
    let it: i32 = sim_int_par.it;

    // Pressure
    var id_x_i: i32 = -get_idx_ff(last);
    var id_x_f: i32 = sim_int_par.x_sz - get_idx_if(last);
    var id_y_i: i32 = -get_idx_ff(last);
    var id_y_f: i32 = sim_int_par.y_sz - get_idx_if(last);
    if(x >= id_x_i && x < id_x_f && y >= id_y_i && y < id_y_f) {
        var vdpxx_dx: f32 = 0.0;
        var vdpyy_dy: f32 = 0.0;
        for(var c: i32 = (ord * 2) - 1; c > -1; c--) {
            let off: i32 = (ord - 1) - c;
            vdpxx_dx += -get_fdc(c) * get_dpx_dx(x + off, y) * one_dx;
            vdpyy_dy += -get_fdc(c) * get_dpy_dy(x, y + off) * one_dy;
        }

        var mdpxx_dx_new: f32 = get_b_x(x - offset) * get_mdpxx_dx(x, y) + get_a_x(x - offset) * vdpxx_dx;
        var mdpyy_dy_new: f32 = get_b_y(y - offset) * get_mdpyy_dy(x, y) + get_a_y(y - offset) * vdpyy_dy;

        vdpxx_dx = vdpxx_dx/get_k_x(x - offset) + mdpxx_dx_new;
        vdpyy_dy = vdpyy_dy/get_k_y(y - offset) + mdpyy_dy_new;

        set_mdpxx_dx(x, y, mdpxx_dx_new);
        set_mdpyy_dy(x, y, mdpyy_dy_new);

        // Atualiza o campo de pressao futuro a partir do passado e do presente
        var press_ft: f32 = 2.0 * get_press_pr(x, y) - get_press_past(x, y) + 
                            dt * dt * (vdpxx_dx + vdpyy_dy) * get_kappa(x, y);

        // Adiciona a fonte
        let idx_src_term: i32 = get_idx_source_term(x, y);
        if(idx_src_term != -1) {
            press_ft += get_source_term(it, idx_src_term) * dt * dt * one_dx * one_dy;
        }
        set_press_ft(x, y, press_ft);
    }
    else {
        // Apply Dirichlet conditions
        set_press_ft(x, y, 0.0);
    }

    // Compute pressure norm L2
    let p_2_old: f32 = p_2;
    let p2: f32 = abs(get_press_ft(x, y));
    p_2 = max(p_2_old, p2);

    // Swap dos valores novos de pressao para valores antigos
    set_press_past(x, y, get_press_pr(x, y));
    set_press_pr(x, y, get_press_ft(x, y));

    // Store sensors velocities
    let sensor: i32 = get_idx_sensor(x, y);
    if(sensor != -1 && it >= get_delay_rec(sensor)) {
        let value_sens_pressure: f32 = get_sens_pressure(it, sensor) + get_press_pr(x, y);
        set_sens_pressure(it, sensor, value_sens_pressure);
    }
}

// Kernel to increase time iteraction [it]
@compute
@workgroup_size(1)
fn incr_it_kernel() {
    sim_int_par.it += 1;
}
