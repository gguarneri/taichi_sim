from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='laplacian_cuda',
    ext_modules=[
        CUDAExtension(
            name='laplacian_cuda',
            sources=['laplacian_kernel.cu'],
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)