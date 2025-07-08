// laplacian_kernel.cu

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void laplacian_kernel_2d(
    const float* __restrict__ u,
    float* __restrict__ out,
    int Nx, int Nz) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= 4 && ix < Nx - 4 && iz >= 4 && iz < Nz - 4) {
        int idx = ix * Nz + iz;

        float d2x = (-1.0/560*u[(ix-4)*Nz + iz] + 8.0/315*u[(ix-3)*Nz + iz]
                    -1.0/5*u[(ix-2)*Nz + iz] + 8.0/5*u[(ix-1)*Nz + iz]
                    -205.0/72*u[idx]
                    +8.0/5*u[(ix+1)*Nz + iz] -1.0/5*u[(ix+2)*Nz + iz]
                    +8.0/315*u[(ix+3)*Nz + iz] -1.0/560*u[(ix+4)*Nz + iz]);

        float d2z = (-1.0/560*u[ix*Nz + (iz-4)] + 8.0/315*u[ix*Nz + (iz-3)]
                    -1.0/5*u[ix*Nz + (iz-2)] + 8.0/5*u[ix*Nz + (iz-1)]
                    -205.0/72*u[idx]
                    +8.0/5*u[ix*Nz + (iz+1)] -1.0/5*u[ix*Nz + (iz+2)]
                    +8.0/315*u[ix*Nz + (iz+3)] -1.0/560*u[ix*Nz + (iz+4)]);

        out[idx] = (d2x + d2z);
    }
}

torch::Tensor laplacian(torch::Tensor u) {
    const auto Nx = u.size(0);
    const auto Nz = u.size(1);
    auto out = torch::zeros_like(u);

    const dim3 threads(16, 16);
    const dim3 blocks((Nx + 15)/16, (Nz + 15)/16);

    laplacian_kernel_2d<<<blocks, threads>>>(
        u.data_ptr<float>(), out.data_ptr<float>(),
        Nx, Nz
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("laplacian", &laplacian, "9-point Laplacian kernel (CUDA)");
}
