// i -> linha, j -> coluna
#define ij(i, j, i_max, j_max) ((((i) >= 0) && ((i) < (i_max)) && ((j) >= 0) && ((j) < (j_max)))?(((i) * (j_max)) + (j)):(-1))

// x -> linha, y -> coluna
#define xy(x, y, x_max, y_max) ((((x) >= 0) && ((x) < (x_max)) && ((y) >= 0) && ((x) < (x_max)))?(((x) * (y_max)) + (y)):(-1))


extern "C" __global__
void test_kernel(float *vx, float *vy, int *idx_fd,
        float dt, float one_dx, float one_dy, int nx, int ny, int _ord)
{
    // Compute column and row indices.
    const int x = blockIdx.x * blockDim.x + threadIdx.x;  // linha
    const int y = blockIdx.y * blockDim.y + threadIdx.y;  // coluna
    
    const int last = _ord - 1;
    const int offset = _ord - 1;
    const int id_x_i = -idx_fd[ij(last, 2, _ord, 4)];
    const int id_x_f = nx - idx_fd[ij(last, 0, _ord, 4)];
    const int id_y_i = -idx_fd[ij(last, 3, _ord, 4)];
    const int id_y_f = ny - idx_fd[ij(last, 1, _ord, 4)];
    const int idx = xy(x, y, nx, ny);

    // Check if within image bounds.
    if (idx == -1)
        return;

    if(x >= id_x_i && x < id_x_f && y >= id_y_i && y < id_y_f) {
        const int c = 1;
        const int idx_xi = idx_fd[ij(c, 0, _ord, 4)];
        const int idx_xf = idx_fd[ij(c, 2, _ord, 4)];
        const int idx_yi = idx_fd[ij(c, 1, _ord, 4)];
        const int idx_yf = idx_fd[ij(c, 3, _ord, 4)];
        
        vx[idx] = x + idx_xi * 1.0f;
        vy[idx] = x + idx_xf * 1.0f;
    }
}

extern "C" __global__
void pressure_kernel(
        float *vx, float *vy, float *pressure, float *kappa_unrelaxed,
        float *memory_dvx_dx, float *memory_dvy_dy, float *value_dvx_dx, float *value_dvy_dy,
        float *a_x_half, float *b_x_half, float *k_x_half,
        float *a_y, float *b_y, float *k_y,
        float *coefs, int *idx_fd, int *idx_src_gpu, float *d_source_term, int it, int nt, int n_pto_src,
        float dt, float one_dx, float one_dy, int nx, int ny, int _ord, float *pressure_l2_norm_gpu)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        const int last = _ord - 1;
        const int offset = _ord - 1;
        const int id_x_i = -idx_fd[ij(last, 2, _ord, 4)];
        const int id_x_f = nx - idx_fd[ij(last, 0, _ord, 4)];
        const int id_y_i = -idx_fd[ij(last, 3, _ord, 4)];
        const int id_y_f = ny - idx_fd[ij(last, 1, _ord, 4)];
        const int idx = xy(x, y, nx, ny);

        if(idx == -1)
            return;
        
        pressure_l2_norm_gpu[0] = 0.0f;
        if(x >= id_x_i && x < id_x_f && y >= id_y_i && y < id_y_f) {
            float dvx_dx = 0.f;
            float dvy_dy = 0.f;

            for (int c = 0; c < _ord; c++) {
                const int idx_xi = idx_fd[ij(c, 0, _ord, 4)];
                const int idx_xf = idx_fd[ij(c, 2, _ord, 4)];
                const int idx_yi = idx_fd[ij(c, 1, _ord, 4)];
                const int idx_yf = idx_fd[ij(c, 3, _ord, 4)];

                dvx_dx += coefs[c] * (vx[xy(x + idx_xi, y, nx, ny)] - vx[xy(x + idx_xf, y, nx, ny)]) * one_dx;
                dvy_dy += coefs[c] * (vy[xy(x, y + idx_yi, nx, ny)] - vy[xy(x, y + idx_yf, nx, ny)]) * one_dy;
            }

            float mdvx_dx_new = b_x_half[x - offset] * memory_dvx_dx[idx] + a_x_half[x - offset] * dvx_dx;
            float mdvy_dy_new = b_y[y - offset] * memory_dvy_dy[idx] + a_y[y - offset] * dvy_dy;

            memory_dvx_dx[idx] = mdvx_dx_new;
            memory_dvy_dy[idx] = mdvy_dy_new;

            dvx_dx = dvx_dx / k_x_half[x - offset] + mdvx_dx_new;
            dvy_dy = dvy_dy / k_y[y - offset] + mdvy_dy_new;

            value_dvx_dx[idx] = dvx_dx;
            value_dvy_dy[idx] = dvy_dy;

            float pressure_new = pressure[idx] + kappa_unrelaxed[idx] * (dvx_dx + dvy_dy) * dt;

            // Adiciona o sinal de fonte, se o pixel fizer parte de uma fonte
            const int idx_src = idx_src_gpu[idx];
            if (idx_src != -1) {
                const int idx_source_term = xy(it - 1, idx_src, nt, n_pto_src);
                pressure_new += d_source_term[idx_source_term] * dt * one_dx * one_dy;
            }
            pressure[idx] = pressure_new;
        }
    }

    extern "C" __global__
    void velocity_kernel(
        float *vx, float *vy, float *pressure, float *rho_grid_vx, float *rho_grid_vy,
        float *memory_dpressure_dx, float *value_dpressure_dx,
        float *memory_dpressure_dy, float *value_dpressure_dy,
        float *a_x, float *b_x, float *k_x,
        float *a_y_half, float *b_y_half, float *k_y_half,
        float *coefs, int *idx_fd, float *sens_pressure, int *idx_sen_gpu, int *delay_rec, int it, int nt, int n_pto_rec,
        float dt, float one_dx, float one_dy, int nx, int ny, int _ord, float* pressure_l2_norm_gpu)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        const int last = _ord - 1;
        const int offset = _ord - 1;
        const int idx = xy(x, y, nx, ny);
        int id_x_i = -idx_fd[ij(last, 3, _ord, 4)];
        int id_x_f = nx - idx_fd[ij(last, 1, _ord, 4)];
        int id_y_i = -idx_fd[ij(last, 3, _ord, 4)];
        int id_y_f = ny - idx_fd[ij(last, 1, _ord, 4)];

        if(idx == -1)
            return;

        // Atualiza Vx
        if(x >= id_x_i && x < id_x_f && y >= id_y_i && y < id_y_f) {
            float dpressure_dx = 0.0f;

            for (int c = 0; c < _ord; c++) {
                const int idx_xi = idx_fd[ij(c, 1, _ord, 4)];
                const int idx_xf = idx_fd[ij(c, 3, _ord, 4)];

                dpressure_dx += coefs[c] * (pressure[xy(x + idx_xi, y, nx, ny)] - pressure[xy(x + idx_xf, y, nx, ny)]) * one_dx;
            }

            float mdpressure_dx_new = b_x[x - offset] * memory_dpressure_dx[idx] + a_x[x - offset] * dpressure_dx;
            memory_dpressure_dx[idx] = mdpressure_dx_new;

            dpressure_dx = dpressure_dx / k_x[x - offset] + mdpressure_dx_new;
            value_dpressure_dx[idx] = dpressure_dx;

            float vx_new = vx[idx] + dt * (dpressure_dx / rho_grid_vx[idx]);
            vx[idx] = vx_new;
        }
        else {
            // Condicao de contorno Dirichlet
            vx[idx] = 0.0f;
        }

        // Atualiza Vy
        id_x_i = -idx_fd[ij(last, 2, _ord, 4)];
        id_x_f = nx - idx_fd[ij(last, 0, _ord, 4)];
        id_y_i = -idx_fd[ij(last, 2, _ord, 4)];
        id_y_f = ny - idx_fd[ij(last, 0, _ord, 4)];

        if(x >= id_x_i && x < id_x_f && y >= id_y_i && y < id_y_f) {
            float dpressure_dy = 0.0f;

            for (int c = 0; c < _ord; c++) {
                const int idx_yi = idx_fd[ij(c, 0, _ord, 4)];
                const int idx_yf = idx_fd[ij(c, 2, _ord, 4)];

                dpressure_dy += coefs[c] * (pressure[xy(x, y + idx_yi, nx, ny)] - pressure[xy(x, y + idx_yf, nx, ny)]) * one_dy;
            }

            float mdpressure_dy_new = b_y_half[y - offset] * memory_dpressure_dy[idx] +
                                      a_y_half[y - offset] * dpressure_dy;
            memory_dpressure_dy[idx]=  mdpressure_dy_new;

            dpressure_dy = dpressure_dy / k_y_half[y-offset] + mdpressure_dy_new;
            value_dpressure_dy[idx] = dpressure_dy;

            float vy_new = vy[idx] + dt * (dpressure_dy / rho_grid_vy[idx]);
            vy[idx] =  vy_new;
        }
        else {
            // Condicao de contorno Dirichlet
            vy[idx] = 0.0f;
        }

        // Calcula a norma L2 da pressao
        float p_2_old = pressure_l2_norm_gpu[0];
        float p_2_new = abs(pressure[idx]);
        pressure_l2_norm_gpu[0] = p_2_old < p_2_new ? p_2_new : p_2_old;
                
        // Armazena o sinal do sensor, se o pixel fizer parte de um receptor
        const int idx_sen = idx_sen_gpu[idx];
        if(idx_sen != -1 and it >= delay_rec[idx_sen]) {
            const int sample = xy(it - 1, idx_sen, nt, n_pto_rec);
            sens_pressure[sample] += pressure[idx];
        }
    }