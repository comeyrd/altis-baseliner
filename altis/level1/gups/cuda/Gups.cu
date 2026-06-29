#include "../Gups.hpp"
#include <baseliner/Register.hpp>
#include <baseliner/core/hardware/cuda/CudaBackend.hpp>

#define POLY 0x0000000000000007ULL

union benchtype {
  uint64_t u64;
  uint2 u32;
};

static __constant__ uint64_t c_m2[64];

static __device__ uint32_t d_error[1];

static __global__ void d_init(size_t n, benchtype *t) {
  for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    t[i].u64 = i;
  }
}

static __device__ uint64_t d_starts(size_t n) {
  if (n == 0) {
    return 1;
  }

  int i = 63 - __clzll(n);

  uint64_t ran = 2;
  while (i > 0) {
    uint64_t temp = 0;
    for (int j = 0; j < 64; j++) {
      if ((ran >> j) & 1) {
        temp ^= c_m2[j];
      }
    }
    ran = temp;
    i -= 1;
    if ((n >> i) & 1) {
      ran = (ran << 1) ^ ((int64_t)ran < 0 ? POLY : 0);
    }
  }

  return ran;
}

enum atomictype_t {
  ATOMICTYPE_CAS,
  ATOMICTYPE_XOR,
};

template <atomictype_t ATOMICTYPE>
__global__ void d_bench(size_t n, benchtype *t) {
  size_t num_threads = gridDim.x * blockDim.x;
  size_t thread_num = blockIdx.x * blockDim.x + threadIdx.x;
  size_t start = thread_num * 4 * n / num_threads;
  size_t end = (thread_num + 1) * 4 * n / num_threads;
  benchtype ran;
  ran.u64 = d_starts(start);
  for (ptrdiff_t i = start; i < end; ++i) {
    ran.u64 = (ran.u64 << 1) ^ ((int64_t)ran.u64 < 0 ? POLY : 0);
    switch (ATOMICTYPE) {
    case ATOMICTYPE_CAS:
      unsigned long long int *address, old, assumed;
      address = (unsigned long long int *)&t[ran.u64 & (n - 1)].u64;
      old = *address;
      do {
        assumed = old;
        old = atomicCAS(address, assumed, assumed ^ ran.u64);
      } while (assumed != old);
      break;
    case ATOMICTYPE_XOR:
      atomicXor(&t[ran.u64 & (n - 1)].u32.x, ran.u32.x);
      atomicXor(&t[ran.u64 & (n - 1)].u32.y, ran.u32.y);
      break;
    }
  }
}

static __global__ void d_check(size_t n, benchtype *t) {
  for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    if (t[i].u64 != i) {
      atomicAdd(d_error, 1);
    }
  }
}

template <>
void GupsWorkload<Baseliner::Hardware::CudaBackend>::setup_device(typename backend::stream_t stream) {
  CHECK_CUDA(cudaMemcpyToSymbol(c_m2, m_m2, sizeof(m_m2)));

  CHECK_CUDA(cudaMallocAsync(&m_d_t, m_n * sizeof(benchtype), stream));

  int device;
  CHECK_CUDA(cudaGetDevice(&device));
  cudaDeviceProp prop;
  CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

  m_grid_size = prop.multiProcessorCount * (prop.maxThreadsPerMultiProcessor / prop.warpSize);
  m_thread_size = prop.warpSize;
  std::cout << "Warp size " << prop.warpSize << "\n";
  std::cout << "Multi-processor count " << prop.multiProcessorCount << "\n";
  std::cout << "Max threads per multi-processor " << prop.maxThreadsPerMultiProcessor << "\n";
}

template <>
void GupsWorkload<Baseliner::Hardware::CudaBackend>::reset_device(typename backend::stream_t stream) {
  d_init<<<m_grid_size, m_thread_size, 0, stream>>>(m_n, static_cast<benchtype *>(m_d_t));
}

template <>
void GupsWorkload<Baseliner::Hardware::CudaBackend>::fetch_results(typename backend::stream_t stream) {
  void *p_error;
  CHECK_CUDA(cudaGetSymbolAddress(&p_error, d_error));
  CHECK_CUDA(cudaMemsetAsync(p_error, 0, sizeof(uint32_t), stream));

  d_check<<<m_grid_size, m_thread_size, 0, stream>>>(m_n, static_cast<benchtype *>(m_d_t));

  CHECK_CUDA(cudaMemcpyAsync(&m_h_error, p_error, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));

  CHECK_CUDA(cudaFree(m_d_t));
  m_d_t = nullptr;
}
template <atomictype_t ATOMICTYPE>
auto atomic_to_str() -> std::string;

template <>
auto atomic_to_str<ATOMICTYPE_CAS>() -> std::string {
  return "CAS";
}
template <>
auto atomic_to_str<ATOMICTYPE_XOR>() -> std::string {
  return "XOR";
}

template <atomictype_t ATOMICTYPE>
class GupsCuda : public GupsWorkload<Baseliner::Hardware::CudaBackend> {
public:
  auto run(typename backend::stream_t stream) -> typename backend::launch_result_t override {
    d_bench<ATOMICTYPE><<<m_grid_size, m_thread_size, 0, stream>>>(m_n, m_d_t_2);
    return {};
  };
  auto specialization() -> std::string override {
    return atomic_to_str<ATOMICTYPE>();
  }
  void setup_device(typename backend::stream_t stream) override {
    GupsWorkload<Baseliner::Hardware::CudaBackend>::setup_device(stream);
    m_d_t_2 = static_cast<benchtype *>(m_d_t);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
  }

private:
  benchtype *m_d_t_2;
  cudaEvent_t start, stop;
};

namespace {
  using GupsCudaCAS = GupsCuda<ATOMICTYPE_CAS>;
  using GupsCudaXOR = GupsCuda<ATOMICTYPE_XOR>;
  BASELINER_REGISTER_WORKLOAD(GupsCudaCAS);
  BASELINER_REGISTER_WORKLOAD(GupsCudaXOR);
} // namespace