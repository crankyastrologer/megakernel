#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <cstdio>

namespace cg = cooperative_groups;

__global__ void coop_test(int *out) {
  cg::grid_group grid = cg::this_grid();
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = 0; i < 1000; ++i) {
    atomicAdd(out + tid, 1);
  }
  grid.sync();
  if (tid == 0) {
    out[0] = 1;
  }
}

int main() {
  int dev = 0;
  cudaSetDevice(dev);
  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, dev);

  printf("device: %s\n", prop.name);
  printf("sm_count: %d\n", prop.multiProcessorCount);
  printf("cc: %d.%d\n", prop.major, prop.minor);
  printf("cooperativeLaunch: %d\n", prop.cooperativeLaunch);
  if (!prop.cooperativeLaunch) {
    printf("COOP: FAIL (device does not support cooperative launch)\n");
    return 1;
  }

  int threads = 256;
  int blocks_per_sm = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, coop_test, threads, 0);
  int blocks = blocks_per_sm * prop.multiProcessorCount;
  printf("blocks_per_sm: %d, grid: %d, threads: %d\n", blocks_per_sm, blocks, threads);
  if (blocks <= 0) {
    printf("COOP: FAIL (no launchable blocks)\n");
    return 1;
  }

  size_t n = (size_t)blocks * threads;
  int *d_out = nullptr;
  cudaMalloc(&d_out, n * sizeof(int));
  cudaMemset(d_out, 0, n * sizeof(int));

  void *args[] = {&d_out};
  cudaError_t err = cudaLaunchCooperativeKernel((void *)coop_test, dim3(blocks), dim3(threads),
                                                args, 0, nullptr);
  if (err != cudaSuccess) {
    printf("launch error: %s\n", cudaGetErrorString(err));
    printf("COOP: FAIL\n");
    cudaFree(d_out);
    return 1;
  }

  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("sync error: %s\n", cudaGetErrorString(err));
    printf("COOP: FAIL\n");
    cudaFree(d_out);
    return 1;
  }

  int first = 0;
  int last = 0;
  cudaMemcpy(&first, d_out, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&last, d_out + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d_out);

  printf("out[0]: %d (expect 1)\n", first);
  printf("out[n-1]: %d (expect 1000)\n", last);
  int ok = (first == 1 && last == 1000);
  printf("COOP: %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
