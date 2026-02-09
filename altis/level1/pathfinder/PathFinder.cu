#include "PathFinder.hpp"
#include <iostream>
#include <random>

#define BLOCK_SIZE 512
#define STR_SIZE 256
#define HALO 1
#define IN_RANGE(x, min, max) ((x) >= (min) && (x) <= (max))
#define CLAMP_RANGE(x, min, max) x = (x < (min)) ? min : ((x > (max)) ? max : x)
#define MIN(a, b) ((a) <= (b) ? (a) : (b))

using namespace Baseliner;

// =================================================================================
// KERNEL CODE (Unchanged Logic)
// =================================================================================

__global__ void dynproc_kernel(int iteration, int *gpuWall, int *gpuSrc, int *gpuResults, int cols, int rows,
                               int startStep, int border) {
  __shared__ int prev[BLOCK_SIZE];
  __shared__ int result[BLOCK_SIZE];

  int bx = blockIdx.x;
  int tx = threadIdx.x;

  // each block finally computes result for a small block
  // after N iterations.
  // it is the non-overlapping small blocks that cover
  // all the input data

  // calculate the small block size
  int small_block_cols = BLOCK_SIZE - iteration * HALO * 2;

  // calculate the boundary for the block according to
  // the boundary of its small block
  int blkX = (int)small_block_cols * (int)bx - (int)border;
  int blkXmax = blkX + (int)BLOCK_SIZE - 1;

  // calculate the global thread coordination
  int xidx = blkX + (int)tx;

  // effective range within this block that falls within
  // the valid range of the input data
  // used to rule out computation outside the boundary.
  int validXmin = (blkX < 0) ? -blkX : 0;
  int validXmax = (blkXmax > (int)cols - 1) ? (int)BLOCK_SIZE - 1 - (blkXmax - (int)cols + 1) : (int)BLOCK_SIZE - 1;

  int W = tx - 1;
  int E = tx + 1;

  W = (W < validXmin) ? validXmin : W;
  E = (E > validXmax) ? validXmax : E;

  bool isValid = IN_RANGE(tx, validXmin, validXmax);

  if (IN_RANGE(xidx, 0, (int)cols - 1)) {
    prev[tx] = gpuSrc[xidx];
  }
  __syncthreads(); // [Ronny] Added sync to avoid race on prev Aug. 14 2012
  bool computed;
  for (int i = 0; i < iteration; i++) {
    computed = false;
    if (IN_RANGE(tx, i + 1, BLOCK_SIZE - i - 2) && isValid) {
      computed = true;
      int left = prev[W];
      int up = prev[tx];
      int right = prev[E];
      int shortest = MIN(left, up);
      shortest = MIN(shortest, right);
      int index = cols * (startStep + i) + xidx;
      result[tx] = shortest + gpuWall[index];
    }
    __syncthreads();
    if (i == iteration - 1)
      break;
    if (computed) // Assign the computation range
      prev[tx] = result[tx];
    __syncthreads(); // [Ronny] Added sync to avoid race on prev Aug. 14 2012
  }

  // update the global memory
  // after the last iteration, only threads coordinated within the
  // small block perform the calculation and switch on ``computed''
  if (computed) {
    gpuResults[xidx] = result[tx];
  }
}

// =================================================================================
// CLASS IMPLEMENTATION
// =================================================================================

void PathfinderInput::generate_random() {
  std::default_random_engine gen(seed);
  std::uniform_int_distribution<int> dist(0, 9);
  for (size_t i = 0; i < m_flat_data.size(); i++) {
    m_flat_data[i] = dist(gen);
  }
}

void PathfinderKernel::cpu(PathfinderOutput &output) {
  // TODO
}

void PathfinderKernel::setup() {
  int rows = m_input.m_rows;
  int cols = m_input.m_cols;
  int size = rows * cols;

  // Allocate device memory
  CHECK_CUDA(cudaMalloc((void **)&m_d_gpuResult[0], sizeof(int) * cols));
  CHECK_CUDA(cudaMalloc((void **)&m_d_gpuResult[1], sizeof(int) * cols));
  CHECK_CUDA(cudaMalloc((void **)&m_d_gpuWall, sizeof(int) * (size - cols)));

  // Copy Input Data
  // m_flat_data structure: [Row 0 (Initial Result)] [Row 1...N (Wall)]
  // Copy first row to gpuResult[0]
  CHECK_CUDA(cudaMemcpy(m_d_gpuResult[0], m_input.m_flat_data.data(), sizeof(int) * cols, cudaMemcpyHostToDevice));

  // Copy remaining rows to gpuWall
  CHECK_CUDA(
      cudaMemcpy(m_d_gpuWall, m_input.m_flat_data.data() + cols, sizeof(int) * (size - cols), cudaMemcpyHostToDevice));
}

void PathfinderKernel::reset() {
  // The kernel swaps between gpuResult[0] and gpuResult[1].
  // If we re-run, we must ensure gpuResult[0] contains the initial seed data again.
  int cols = m_input.m_cols;
  CHECK_CUDA(cudaMemcpy(m_d_gpuResult[0], m_input.m_flat_data.data(), sizeof(int) * cols, cudaMemcpyHostToDevice));
}

void PathfinderKernel::run(std::shared_ptr<cudaStream_t> &stream) {
  int rows = m_input.m_rows;
  int cols = m_input.m_cols;
  int pyramid_height = m_input.m_pyramid_height;

  // Calculate Grid/Block parameters
  int borderCols = (pyramid_height)*HALO;
  int smallBlockCol = BLOCK_SIZE - (pyramid_height)*HALO * 2;
  int blockCols = cols / smallBlockCol + ((cols % smallBlockCol == 0) ? 0 : 1);

  dim3 dimBlock(BLOCK_SIZE);
  dim3 dimGrid(blockCols);

  int src = 1;
  int dst = 0;

  for (int t = 0; t < rows - 1; t += pyramid_height) {
    int temp = src;
    src = dst;
    dst = temp;

    dynproc_kernel<<<dimGrid, dimBlock, 0, *stream>>>(MIN(pyramid_height, rows - t - 1), m_d_gpuWall,
                                                      m_d_gpuResult[src], m_d_gpuResult[dst], cols, rows, t,
                                                      borderCols);
  }

  // Store which buffer has the final result for teardown
  m_final_ret_idx = dst;
}

void PathfinderKernel::teardown(PathfinderOutput &output) {
  // Copy the final result from the correct buffer
  CHECK_CUDA(cudaMemcpy(output.m_result_host.data(), m_d_gpuResult[m_final_ret_idx], sizeof(int) * m_input.m_cols,
                        cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(m_d_gpuResult[0]));
  CHECK_CUDA(cudaFree(m_d_gpuResult[1]));
  CHECK_CUDA(cudaFree(m_d_gpuWall));
}