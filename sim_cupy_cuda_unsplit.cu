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
void pressure_first_der_kernel(
        float *press_pr, float *rho_x, float *rho_y,
        float *dpx_dx, float *dpy_dy, 
        float *mdpx_dx, float *mdpy_dy,
        float *a_x_h, float *b_x_h, float *k_x_h,
        float *a_y_h, float *b_y_h, float *k_y_h,
        float *coefs, int *idx_fd,
        float one_dx, float one_dy, int nx, int ny, int _ord)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        const int last = _ord - 1;
        const int offset = _ord - 1;
        const int id_x_i = -idx_fd[ij(last, 2, _ord, 4)];
        const int id_x_f = nx - idx_fd[ij(last, 0, _ord, 4)];
        const int id_y_i = -idx_fd[ij(last, 2, _ord, 4)];
        const int id_y_f = ny - idx_fd[ij(last, 0, _ord, 4)];
        const int idx = xy(x, y, nx, ny);

        if(idx == -1)
            return;
        
        if(x >= id_x_i && x < id_x_f && y >= id_y_i && y < id_y_f) {
            float vdpx_dx = 0.f;
            float vdpy_dy = 0.f;

            for (int c = 0; c < _ord * 2; c++) {
                int off = c - (_ord - 1);
                int idx_x = xy(x + off, y, nx, ny);
                int idx_y = xy(x, y + off, nx, ny);
                vdpx_dx += coefs[c] * press_pr[idx_x] * one_dx;
                vdpy_dy += coefs[c] * press_pr[idx_y] * one_dy;
            }

            float mdpx_dx_new = b_x_h[x - offset] * mdpx_dx[idx] + a_x_h[x - offset] * vdpx_dx;
            float mdpy_dy_new = b_y_h[y - offset] * mdpy_dy[idx] + a_y_h[y - offset] * vdpy_dy;

            mdpx_dx[idx] = mdpx_dx_new;
            mdpy_dy[idx] = mdpy_dy_new;

            vdpx_dx = vdpx_dx / k_x_h[x - offset] + mdpx_dx_new;
            vdpy_dy = vdpy_dy / k_y_h[y - offset] + mdpy_dy_new;

            dpx_dx[idx] = vdpx_dx / rho_x[idx];
            dpy_dy[idx] = vdpy_dy / rho_y[idx];
        }
    }


extern "C" __global__
void pressure_second_der_kernel(
        float *press_past, float *press_pr, float *press_ft, float *kappa,
        float *dpx_dx, float *dpy_dy, 
        float *mdpxx_dx, float *mdpyy_dy,
        float *a_x, float *b_x, float *k_x,
        float *a_y, float *b_y, float *k_y,
        float *coefs, int *idx_fd,
        float dt, float one_dx, float one_dy, int nx, int ny, int _ord)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        const int last = _ord - 1;
        const int offset = _ord - 1;
        const int id_x_i = -idx_fd[ij(last, 3, _ord, 4)];
        const int id_x_f = nx - idx_fd[ij(last, 1, _ord, 4)];
        const int id_y_i = -idx_fd[ij(last, 3, _ord, 4)];
        const int id_y_f = ny - idx_fd[ij(last, 1, _ord, 4)];
        const int idx = xy(x, y, nx, ny);

        if(idx == -1)
            return;

        if(x >= id_x_i && x < id_x_f && y >= id_y_i && y < id_y_f) {
            float vdpxx_dx = 0.f;
            float vdpyy_dy = 0.f;

            for (int c = (_ord * 2) - 1; c > -1; --c) {
                int off = (_ord - 1) - c;
                int idx_x = xy(x + off, y, nx, ny);
                int idx_y = xy(x, y + off, nx, ny);
                vdpxx_dx += -coefs[c] * dpx_dx[idx_x] * one_dx;
                vdpyy_dy += -coefs[c] * dpy_dy[idx_y] * one_dy;
            }

            float mdpxx_dx_new = b_x[x - offset] * mdpxx_dx[idx] + a_x[x - offset] * vdpxx_dx;
            float mdpyy_dy_new = b_y[y - offset] * mdpyy_dy[idx] + a_y[y - offset] * vdpyy_dy;

            mdpxx_dx[idx] = mdpxx_dx_new;
            mdpyy_dy[idx] = mdpyy_dy_new;

            vdpxx_dx = vdpxx_dx / k_x[x - offset] + mdpxx_dx_new;
            vdpyy_dy = vdpyy_dy / k_y[y - offset] + mdpyy_dy_new;

            // Atualiza o campo de pressao futuro a partir do passado e do presente
            press_ft[idx] = 2.0 * press_pr[idx] - press_past[idx] + dt*dt * (vdpxx_dx + vdpyy_dy) * kappa[idx];
        }
    }


    extern "C" __global__
    void dirichlet_boundary_kernel(float *press_past, float *press_pr, float *press_ft,
                                   int *idx_fd, int nx, int ny, int _ord) {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        
        const int last = _ord - 1;
        const int offset = _ord - 1;
        const int id_x_i = -idx_fd[ij(last, 2, _ord, 4)];
        const int id_x_f = nx - idx_fd[ij(last, 0, _ord, 4)];
        const int id_y_i = -idx_fd[ij(last, 2, _ord, 4)];
        const int id_y_f = ny - idx_fd[ij(last, 0, _ord, 4)];
        const int idx = xy(x, y, nx, ny);

        if(idx == -1)
            return;

        if(x < id_x_i || x > id_x_f || y < id_y_i || y > id_y_f) {
            press_ft[idx] = 0.0f;
        }

        // Swap dos valores novos de pressao para valores antigos
        press_past[idx] = press_pr[idx];
        press_pr[idx] = press_ft[idx];
    }