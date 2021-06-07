#include <torch/extension.h>
#include "communicator.h"


Communicator::Communicator(int rank, int size):m_rank(rank), m_size(size) {
    m_current_comm = 0;
    m_num_comms = 1;

    m_nccl_ids = new ncclUniqueId[1000];
    m_streams = new cudaStream_t[1000];
    m_nccl_comms = new ncclComm_t[1000];

    for (int i = 0; i < m_num_comms; i++) {
	    if (rank == 0) ncclGetUniqueId(&m_nccl_ids[i]);
	    MPICHECK(MPI_Bcast((void *)&m_nccl_ids[i], sizeof(m_nccl_ids[i]), MPI_BYTE, 0, MPI_COMM_WORLD));
    }
    //ncclGroupStart();
    for (int i = 0; i < m_num_comms; i++) {
	    CUDACHECK(cudaStreamCreate(&m_streams[i]));
	    NCCLCHECK(ncclCommInitRank(&m_nccl_comms[i], size, m_nccl_ids[i], rank));
    }
    //ncclGroupEnd();

	//CUDACHECK(cudaStreamCreate(&m_stream));
	//NCCLCHECK(ncclCommInitRank(&m_nccl_comm, size, m_nccl_id, rank));
}

Communicator::~Communicator() {
	//ncclCommDestroy(m_nccl_comm);
    for (int i = 0; i < m_num_comms; i++) {
	    NCCLCHECK(ncclCommDestroy(m_nccl_comms[i]));
        cudaStreamDestroy(m_streams[i]);
    }
    delete m_streams;
    delete m_nccl_comms;
}

void Communicator::_extendComms(int n_comms) {
    if (m_num_comms >= n_comms) return;
    for (int i = 0; i < n_comms-m_num_comms; i++) {
	    if (m_rank == 0) ncclGetUniqueId(&m_nccl_ids[i+m_num_comms]);
	    MPICHECK(MPI_Bcast((void *)&m_nccl_ids[i+m_num_comms], sizeof(m_nccl_ids[i+m_num_comms]), MPI_BYTE, 0, MPI_COMM_WORLD));

	    CUDACHECK(cudaStreamCreate(&m_streams[i+m_num_comms]));
	    NCCLCHECK(ncclCommInitRank(&m_nccl_comms[i+m_num_comms], m_size, m_nccl_ids[i+m_num_comms], m_rank));
    }
    m_num_comms = n_comms;
}

void Communicator::synchronize() {
    //CUDACHECK(cudaStreamSynchronize(m_stream));
    for (int i = 0; i < m_num_comms; i++) {
        CUDACHECK(cudaStreamSynchronize(m_streams[i]));
    }
    
}

//void Communicator::allReduce(const void* sendbuff, void* recvbuff, int size) {
//    NCCLCHECK(ncclAllReduce(sendbuff, recvbuff, size, ncclFloat, ncclSum, m_nccl_comm, m_stream));
//}

void Communicator::allReduce(torch::Tensor tensor) {
    NCCLCHECK(ncclAllReduce(tensor.data_ptr<float>(), tensor.data_ptr<float>(), tensor.numel(), ncclFloat, ncclSum, m_nccl_comms[m_current_comm], m_streams[m_current_comm]));
    m_current_comm++;
    m_current_comm %= m_num_comms;
}

void Communicator::reduce(torch::Tensor tensor, int root) {
    NCCLCHECK(ncclReduce(tensor.data_ptr<float>(), tensor.data_ptr<float>(), tensor.numel(), ncclFloat, ncclSum, root, m_nccl_comms[m_current_comm], m_streams[m_current_comm]));
    m_current_comm++;
    m_current_comm %= m_num_comms;
}

//void Communicator::multiBcast(vector<torch::Tensor> &tensor_list, void (*op)(torch::Tensor)) {
void Communicator::multiBcast(vector<torch::Tensor> &tensor_list, vector<torch::Tensor> &output_list, const std::function<void(torch::Tensor, torch::Tensor)> &op) {
    vector<int> tensor_ranks;
    int assigned_rank = 0;
    int num_comm_tensors = 0;
    int min_tensor_size = 512*512;
    for (unsigned i = 0; i < tensor_list.size(); i++) {
        torch::Tensor tensor = tensor_list[i];
        if (tensor.numel() < min_tensor_size) {
            tensor_ranks.push_back(-1);
        } else {
            tensor_ranks.push_back(assigned_rank);
            assigned_rank++;
            assigned_rank %= m_size;
            num_comm_tensors++;
        }
    }
    if (m_size > 1) {
        _extendComms(num_comm_tensors);
    }
	for (unsigned i = 0; i < tensor_list.size(); i++) {
        torch::Tensor tensor = tensor_list[i];
        torch::Tensor output = output_list[i];
        assigned_rank = tensor_ranks[i];
        if (assigned_rank == -1) {
            op(tensor, output);
        } else {
            if (assigned_rank == m_rank) {
                op(tensor, output);
            } 
        }
    }
	for (unsigned i = 0; i < tensor_list.size(); i++) {
        torch::Tensor output = output_list[i];
        assigned_rank = tensor_ranks[i];
        if (m_size > 1 and assigned_rank >= 0) {
            NCCLCHECK(ncclBroadcast(output.data_ptr<float>(), output.data_ptr<float>(), output.numel(), ncclFloat, assigned_rank, m_nccl_comms[m_current_comm], m_streams[m_current_comm])); 
            m_current_comm++;
            m_current_comm %= m_num_comms;
        }
    }


}

