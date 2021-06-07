#ifndef F_COMMUNICATER_H
#define F_COMMUNICATER_H
#include <torch/extension.h>
#include <cstdlib>
#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <vector>
#include "cuda_runtime.h"
#include "nccl.h"
#include "mpi.h"

using namespace std;

#define MPICHECK(cmd) do {                          \
  int e = cmd;                                      \
  if( e != MPI_SUCCESS ) {                          \
    printf("Failed: MPI error %s:%d '%d'\n",        \
        __FILE__,__LINE__, e);   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


class Communicator {
public:
    Communicator(int rank, int size);
    ~Communicator();
	//void allReduce(const void* sendbuff, void* recvbuff, int size);
//std::vector<torch::Tensor> tcmm_symeig(torch::Tensor a) {
	void allReduce(torch::Tensor tensor);
	void reduce(torch::Tensor tensor, int root);
	//void multiBcast(vector<torch::Tensor> &tensor_list, void (*op)(torch::Tensor));
	void multiBcast(vector<torch::Tensor> &tensor_list, vector<torch::Tensor> &output_list, const std::function<void(torch::Tensor, torch::Tensor)> &op);
    void synchronize();
    void _extendComms(int n_comms);
private:
    //ncclUniqueId m_nccl_id;
    //ncclComm_t m_nccl_comm;
	//cudaStream_t m_stream;
    ncclUniqueId* m_nccl_ids;
    ncclComm_t* m_nccl_comms;
	cudaStream_t* m_streams;
    int m_rank;
    int m_size;
    int m_current_comm;
    int m_num_comms;
};

#endif //F_COMMUNICATER_H
