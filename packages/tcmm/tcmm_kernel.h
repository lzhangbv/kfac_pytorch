
#include <torch/extension.h>

std::vector<torch::Tensor> tcmm_symeig(torch::Tensor a);
std::vector<torch::Tensor> tcmm_symeig_sparse(torch::Tensor a);
