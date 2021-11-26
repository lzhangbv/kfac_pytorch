import os
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension, _find_cuda_home

os.system('pip uninstall -y tcmm')
#os.system('make clean; make -j%d' % os.cpu_count())

CUDA_DIR = _find_cuda_home()

#NCCL_DIR= '/home/comp/15485625/downloads/nccl_2.4.8-1+cuda10.1_x86_64'
#MPI_DIR = '/usr/local/openmpi/openmpi-4.0.1'
#NCCL_DIR = '/home/esetstore/downloads/nccl_2.4.7-1+cuda10.1_x86_64'

NCCL_DIR = '/home/esetstore/downloads/nccl_2.6.4-1+cuda10.1_x86_64'
MPI_DIR = '/home/esetstore/.local/openmpi-4.0.1'

# Python interface
setup(
    name='tcmm',
    version='0.2.0',
    install_requires=['torch'],
    packages=['tcmm'],
    package_dir={'tcmm': './'},
    ext_modules=[
        CUDAExtension(
            name='tcmm',
            include_dirs=['./', 
                NCCL_DIR+'/include', 
                MPI_DIR+'/include',
                CUDA_DIR+'/samples/common/inc'],
            sources=[
                'src/communicator.cpp',
                'src/tcmm.cpp',
                'src/tcmm_kernel.cu',
            ],
            libraries=['cusolver', 'nccl', 'mpi', 'cublas'],
            library_dirs=['objs', CUDA_DIR+'/lib64', NCCL_DIR+'/lib', MPI_DIR+'/lib'],
            # extra_compile_args=['-g']
        )
    ],
    cmdclass={'build_ext': BuildExtension},
    author='Shaohuai Shi',
    author_email='shaohuais@cse.ust.hk',
    description='Efficient PyTorch Extension for K-FAC',
    keywords='Pytorch C++ Extension',
    zip_safe=False,
)
