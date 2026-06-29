#include "../PathFinder.hpp"
#include <baseliner/Register.hpp>
#include <baseliner/core/hardware/cuda/CudaBackend.hpp>

#define BLOCK_SIZE 512
#define HALO 1
#define IN_RANGE(x, min, max) ((x) >= (min) && (x) <= (max))
#define MIN(a, b) ((a) <= (b) ? (a) : (b))

__global__ void dynproc_kernel(int iteration, int *gpuWall, int *gpuSrc, int *gpuResults, int cols, int rows,
                               int startStep, int border) {
  __shared__ int prev[BLOCK_SIZE];
  __shared__ int result[BLOCK_SIZE];

  int bx = blockIdx.x;
  int tx = threadIdx.x;

  int small_block_cols = BLOCK_SIZE - iteration * HALO * 2;

  int blkX = (int)small_block_cols * (int)bx - (int)border;
  int blkXmax = blkX + (int)BLOCK_SIZE - 1;

  int xidx = blkX + (int)tx;

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
  __syncthreads();
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
    if (computed)
      prev[tx] = result[tx];
    __syncthreads();
  }

  if (computed) {
    gpuResults[xidx] = result[tx];
  }
}

template <>
void PathfinderWorkload<Baseliner::Hardware::CudaBackend>::setup_device(typename backend::stream_t stream) {
  size_t size = m_rows * m_cols;

  CHECK_CUDA(cudaMallocAsync(&m_d_gpuResult[0], sizeof(int) * m_cols, stream));
  CHECK_CUDA(cudaMallocAsync(&m_d_gpuResult[1], sizeof(int) * m_cols, stream));
  CHECK_CUDA(cudaMallocAsync(&m_d_gpuWall, sizeof(int) * (size - m_cols), stream));

  CHECK_CUDA(cudaMemcpyAsync(m_d_gpuResult[0], m_h_data.data(), sizeof(int) * m_cols, cudaMemcpyHostToDevice, stream));
  CHECK_CUDA(cudaMemcpyAsync(m_d_gpuWall, m_h_data.data() + m_cols, sizeof(int) * (size - m_cols),
                             cudaMemcpyHostToDevice, stream));
}

template <>
void PathfinderWorkload<Baseliner::Hardware::CudaBackend>::reset_device(typename backend::stream_t stream) {
  CHECK_CUDA(cudaMemcpyAsync(m_d_gpuResult[0], m_h_data.data(), sizeof(int) * m_cols, cudaMemcpyHostToDevice, stream));
}

template <>
auto PathfinderWorkload<Baseliner::Hardware::CudaBackend>::run(typename backend::stream_t stream) -> std::monostate {
  int src = 1, dst = 0;
  for (int t = 0; t < m_rows - 1; t += m_pyramid_height) {
    int temp = src;
    src = dst;
    dst = temp;

    int iteration = MIN(m_pyramid_height, m_rows - t - 1);
    dynproc_kernel<<<m_blockCols, BLOCK_SIZE, 0, stream>>>(iteration, m_d_gpuWall, m_d_gpuResult[src],
                                                           m_d_gpuResult[dst], m_cols, m_rows, t, m_borderCols);
  }
  m_final_ret = dst;
  return {};
}

template <>
void PathfinderWorkload<Baseliner::Hardware::CudaBackend>::fetch_results(typename backend::stream_t stream) {
  CHECK_CUDA(cudaMemcpyAsync(m_h_result.data(), m_d_gpuResult[m_final_ret], sizeof(int) * m_cols,
                             cudaMemcpyDeviceToHost, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));

  CHECK_CUDA(cudaFree(m_d_gpuResult[0]));
  CHECK_CUDA(cudaFree(m_d_gpuResult[1]));
  CHECK_CUDA(cudaFree(m_d_gpuWall));

  m_d_gpuResult[0] = nullptr;
  m_d_gpuResult[1] = nullptr;
  m_d_gpuWall = nullptr;
}

namespace {
  using PathFinderCuda = PathfinderWorkload<Baseliner::Hardware::CudaBackend>;
  BASELINER_REGISTER_WORKLOAD(PathFinderCuda);
} // namespace