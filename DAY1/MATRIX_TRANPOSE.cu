#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>


#define N 64
#define EACH_THREAD_HANDLES_X 2
#define EACH_THREAD_HANDLES_Y 2


__global__ void MATRIX_TRANSPOSE(const int *d_MAT, int *d_RES) {
    __shared__ int s_mem[4][4];

    int row = blockIdx.y + blockDim.y + threadIdx.y;
    int col = blockIdx.x + blockDim.x + threadIdx.x;

    for (int i = 0; i < EACH_THREAD_HANDLES_Y; i++) {
        for (int j = 0; j < EACH_THREAD_HANDLES_X; j++) {
            int in_row = row * EACH_THREAD_HANDLES_Y + i;
            int in_col = col * EACH_THREAD_HANDLES_X + j;
            if (in_row < N && in_col < N) {
                s_mem[threadIdx.x+j][threadIdx.y+i]  = d_MAT[in_row * N + in_col];
            }
        }
    }
    __syncthreads();

    for (int dy = 0; dy < EACH_THREAD_HANDLES_Y; dy++) {
        for (int dx = 0; dx < EACH_THREAD_HANDLES_X; dx++) {
            int out_row = col * EACH_THREAD_HANDLES_X + dx;
            int out_col = row * EACH_THREAD_HANDLES_Y + dy;
            if (out_row < N && out_col < N) {
                d_RES[out_row * N + out_col] = s_mem[threadIdx.x * EACH_THREAD_HANDLES_X + dx][threadIdx.y * EACH_THREAD_HANDLES_Y + dy];
            }
        }
    }
}

int main() {
    int *h_MAT = (int*)malloc(sizeof(int) * N * N);
    for (int i=0; i<N; i++) {
        for (int j=0; j<N; j++) {
            h_MAT[i*N + j] = (i+1) * N + j + 1;
        }
    }

    int *d_MAT, *d_RES;
    int size_d = sizeof(int) * N * N;
    cudaError_t err = cudaMalloc((void**)(&d_MAT), size_d);
    if (err != cudaSuccess) {
        std::cout << "Failed to allocate the memory" << std::endl;
        return 1;
    }
    err = cudaMalloc((void**)(&d_RES), size_d);
    if (err != cudaSuccess) {
        std::cout << "Failed to allocate the memory" << std::endl;
        return 1;
    }

    err = cudaMemcpy(d_MAT, h_MAT, size_d, cudaMemcpyHostToDevice);

    int N_THREADS_X = 2;
    int N_THREADS_Y = 2;
    int BLOCKS_X = (N + N_THREADS_X - 1) / N_THREADS_X;
    int BLOCKS_Y = (N + N_THREADS_Y - 1) / N_THREADS_Y;

    dim3 block(N_THREADS_X, N_THREADS_Y);
    dim3 grid(BLOCKS_X, BLOCKS_Y);
    std::cout << BLOCKS_X << " " << BLOCKS_Y << std::endl;

    MATRIX_TRANSPOSE<<<grid, block>>>(d_MAT, d_RES);

    return 0;
}