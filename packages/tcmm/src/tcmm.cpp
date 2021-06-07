#include <torch/extension.h>
#include <pybind11/functional.h>

#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "tcmm_kernel.h"
#include "communicator.h"

namespace py = pybind11;


std::vector<torch::Tensor> f_symeig(torch::Tensor a) {
    auto c = tcmm_symeig(a);
    return c;
}

std::vector<torch::Tensor> f_symeig_sparse(torch::Tensor a) {
    auto c = tcmm_symeig_sparse(a);
    return c;
}

torch::Tensor f_gemm_ex(torch::Tensor a, torch::Tensor b) {
    auto c = tcmm_gemm_ex(a, b);
    return c;
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("f_symeig", &f_symeig, "TCMM: Eigendecomposition using cuSolver");
    m.def("f_symeig_sparse", &f_symeig_sparse, "TCMM: Eigendecomposition using cuSolverSP");
    m.def("f_gemm_ex", &f_gemm_ex, "TCMM: GEMM with Tensor Core using cuBLAS");

    std::string name = std::string("Communicator");
    py::class_<Communicator>(m, name.c_str())
        .def(py::init<int, int>())
        .def("allReduce", &Communicator::allReduce)
        .def("multiBcast", &Communicator::multiBcast)
        .def("reduce", &Communicator::reduce)
        .def("synchronize", &Communicator::synchronize)
        .def("__repr__", [](const Communicator &a) { return "Communicator"; });

}
