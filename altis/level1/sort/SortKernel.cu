#include "SortKernel.hpp"
#include <cmath>
#include <memory>

// =================================================================================
// ORIGINAL KERNEL CODE (Unchanged Logic)
// =================================================================================

#define WARP_SIZE 32
#define SORT_BLOCK_SIZE 128
#define SCAN_BLOCK_SIZE 256
using namespace Baseliner;
typedef unsigned int uint;

__device__ uint scanLSB(const uint val, uint *s_data) {
  int idx = threadIdx.x;
  s_data[idx] = 0;
  __syncthreads();

  idx += blockDim.x;

  uint t;
  s_data[idx] = val;
  __syncthreads();
  t = s_data[idx - 1];
  __syncthreads();
  s_data[idx] += t;
  __syncthreads();
  t = s_data[idx - 2];
  __syncthreads();
  s_data[idx] += t;
  __syncthreads();
  t = s_data[idx - 4];
  __syncthreads();
  s_data[idx] += t;
  __syncthreads();
  t = s_data[idx - 8];
  __syncthreads();
  s_data[idx] += t;
  __syncthreads();
  t = s_data[idx - 16];
  __syncthreads();
  s_data[idx] += t;
  __syncthreads();
  t = s_data[idx - 32];
  __syncthreads();
  s_data[idx] += t;
  __syncthreads();
  t = s_data[idx - 64];
  __syncthreads();
  s_data[idx] += t;
  __syncthreads();

  return s_data[idx] - val;
}

__device__ uint4 scan4(uint4 idata, uint *ptr) {
  uint4 val4 = idata;
  uint4 sum;

  sum.x = val4.x;
  sum.y = val4.y + sum.x;
  sum.z = val4.z + sum.y;
  uint val = val4.w + sum.z;

  val = scanLSB(val, ptr);

  val4.x = val;
  val4.y = val + sum.x;
  val4.z = val + sum.y;
  val4.w = val + sum.z;

  return val4;
}

__global__ void radixSortBlocks(const uint nbits, const uint startbit, uint4 *keysOut, uint4 *valuesOut, uint4 *keysIn,
                                uint4 *valuesIn) {
  __shared__ uint sMem[512];

  const uint i = threadIdx.x + (blockIdx.x * blockDim.x);
  const uint tid = threadIdx.x;
  const uint localSize = blockDim.x;

  uint4 key, value;
  key = keysIn[i];
  value = valuesIn[i];

  for (uint shift = startbit; shift < (startbit + nbits); ++shift) {
    uint4 lsb;
    lsb.x = !((key.x >> shift) & 0x1);
    lsb.y = !((key.y >> shift) & 0x1);
    lsb.z = !((key.z >> shift) & 0x1);
    lsb.w = !((key.w >> shift) & 0x1);

    uint4 address = scan4(lsb, sMem);

    __shared__ uint numtrue;

    if (tid == localSize - 1) {
      numtrue = address.w + lsb.w;
    }
    __syncthreads();

    uint4 rank;
    const int idx = tid * 4;
    rank.x = lsb.x ? address.x : numtrue + idx - address.x;
    rank.y = lsb.y ? address.y : numtrue + idx + 1 - address.y;
    rank.z = lsb.z ? address.z : numtrue + idx + 2 - address.z;
    rank.w = lsb.w ? address.w : numtrue + idx + 3 - address.w;

    sMem[(rank.x & 3) * localSize + (rank.x >> 2)] = key.x;
    sMem[(rank.y & 3) * localSize + (rank.y >> 2)] = key.y;
    sMem[(rank.z & 3) * localSize + (rank.z >> 2)] = key.z;
    sMem[(rank.w & 3) * localSize + (rank.w >> 2)] = key.w;
    __syncthreads();

    key.x = sMem[tid];
    key.y = sMem[tid + localSize];
    key.z = sMem[tid + 2 * localSize];
    key.w = sMem[tid + 3 * localSize];
    __syncthreads();

    sMem[(rank.x & 3) * localSize + (rank.x >> 2)] = value.x;
    sMem[(rank.y & 3) * localSize + (rank.y >> 2)] = value.y;
    sMem[(rank.z & 3) * localSize + (rank.z >> 2)] = value.z;
    sMem[(rank.w & 3) * localSize + (rank.w >> 2)] = value.w;
    __syncthreads();

    value.x = sMem[tid];
    value.y = sMem[tid + localSize];
    value.z = sMem[tid + 2 * localSize];
    value.w = sMem[tid + 3 * localSize];
    __syncthreads();
  }
  keysOut[i] = key;
  valuesOut[i] = value;
}

__global__ void findRadixOffsets(uint2 *keys, uint *counters, uint *blockOffsets, uint startbit, uint numElements,
                                 uint totalBlocks) {
  __shared__ uint sStartPointers[16];
  extern __shared__ uint sRadix1[];

  uint groupId = blockIdx.x;
  uint localId = threadIdx.x;
  uint groupSize = blockDim.x;

  uint2 radix2;
  radix2 = keys[threadIdx.x + (blockIdx.x * blockDim.x)];

  sRadix1[2 * localId] = (radix2.x >> startbit) & 0xF;
  sRadix1[2 * localId + 1] = (radix2.y >> startbit) & 0xF;

  if (localId < 16) {
    sStartPointers[localId] = 0;
  }
  __syncthreads();

  if ((localId > 0) && (sRadix1[localId] != sRadix1[localId - 1])) {
    sStartPointers[sRadix1[localId]] = localId;
  }
  if (sRadix1[localId + groupSize] != sRadix1[localId + groupSize - 1]) {
    sStartPointers[sRadix1[localId + groupSize]] = localId + groupSize;
  }
  __syncthreads();

  if (localId < 16) {
    blockOffsets[groupId * 16 + localId] = sStartPointers[localId];
  }
  __syncthreads();

  if ((localId > 0) && (sRadix1[localId] != sRadix1[localId - 1])) {
    sStartPointers[sRadix1[localId - 1]] = localId - sStartPointers[sRadix1[localId - 1]];
  }
  if (sRadix1[localId + groupSize] != sRadix1[localId + groupSize - 1]) {
    sStartPointers[sRadix1[localId + groupSize - 1]] =
        localId + groupSize - sStartPointers[sRadix1[localId + groupSize - 1]];
  }

  if (localId == groupSize - 1) {
    sStartPointers[sRadix1[2 * groupSize - 1]] = 2 * groupSize - sStartPointers[sRadix1[2 * groupSize - 1]];
  }
  __syncthreads();

  if (localId < 16) {
    counters[localId * totalBlocks + groupId] = sStartPointers[localId];
  }
}

__global__ void reorderData(uint startbit, uint *outKeys, uint *outValues, uint2 *keys, uint2 *values,
                            uint *blockOffsets, uint *offsets, uint *sizes, uint totalBlocks) {
  uint GROUP_SIZE = blockDim.x;
  __shared__ uint2 sKeys2[256];
  __shared__ uint2 sValues2[256];
  __shared__ uint sOffsets[16];
  __shared__ uint sBlockOffsets[16];
  uint *sKeys1 = (uint *)sKeys2;
  uint *sValues1 = (uint *)sValues2;

  uint blockId = blockIdx.x;

  uint i = blockId * blockDim.x + threadIdx.x;

  sKeys2[threadIdx.x] = keys[i];
  sValues2[threadIdx.x] = values[i];

  if (threadIdx.x < 16) {
    sOffsets[threadIdx.x] = offsets[threadIdx.x * totalBlocks + blockId];
    sBlockOffsets[threadIdx.x] = blockOffsets[blockId * 16 + threadIdx.x];
  }
  __syncthreads();

  uint radix = (sKeys1[threadIdx.x] >> startbit) & 0xF;
  uint globalOffset = sOffsets[radix] + threadIdx.x - sBlockOffsets[radix];

  outKeys[globalOffset] = sKeys1[threadIdx.x];
  outValues[globalOffset] = sValues1[threadIdx.x];

  radix = (sKeys1[threadIdx.x + GROUP_SIZE] >> startbit) & 0xF;
  globalOffset = sOffsets[radix] + threadIdx.x + GROUP_SIZE - sBlockOffsets[radix];

  outKeys[globalOffset] = sKeys1[threadIdx.x + GROUP_SIZE];
  outValues[globalOffset] = sValues1[threadIdx.x + GROUP_SIZE];
}

__device__ uint scanLocalMem(const uint val, uint *s_data) {
  int idx = threadIdx.x;
  s_data[idx] = 0.0f;
  __syncthreads();

  idx += blockDim.x;

  uint t;
  s_data[idx] = val;
  __syncthreads();
  t = s_data[idx - 1];
  __syncthreads();
  s_data[idx] += t;
  __syncthreads();
  t = s_data[idx - 2];
  __syncthreads();
  s_data[idx] += t;
  __syncthreads();
  t = s_data[idx - 4];
  __syncthreads();
  s_data[idx] += t;
  __syncthreads();
  t = s_data[idx - 8];
  __syncthreads();
  s_data[idx] += t;
  __syncthreads();
  t = s_data[idx - 16];
  __syncthreads();
  s_data[idx] += t;
  __syncthreads();
  t = s_data[idx - 32];
  __syncthreads();
  s_data[idx] += t;
  __syncthreads();
  t = s_data[idx - 64];
  __syncthreads();
  s_data[idx] += t;
  __syncthreads();
  t = s_data[idx - 128];
  __syncthreads();
  s_data[idx] += t;
  __syncthreads();

  return s_data[idx - 1];
}

__global__ void scan(uint *g_odata, uint *g_idata, uint *g_blockSums, const int n, const bool fullBlock,
                     const bool storeSum) {
  __shared__ uint s_data[512];

  uint4 tempData;
  uint4 threadScanT;
  uint res;
  uint4 *inData = (uint4 *)g_idata;

  const int gid = (blockIdx.x * blockDim.x) + threadIdx.x;
  const int tid = threadIdx.x;
  const int i = gid * 4;

  if (fullBlock || i + 3 < n) {
    tempData = inData[gid];
    threadScanT.x = tempData.x;
    threadScanT.y = tempData.y + threadScanT.x;
    threadScanT.z = tempData.z + threadScanT.y;
    threadScanT.w = tempData.w + threadScanT.z;
    res = threadScanT.w;
  } else {
    threadScanT.x = (i < n) ? g_idata[i] : 0.0f;
    threadScanT.y = ((i + 1 < n) ? g_idata[i + 1] : 0.0f) + threadScanT.x;
    threadScanT.z = ((i + 2 < n) ? g_idata[i + 2] : 0.0f) + threadScanT.y;
    threadScanT.w = ((i + 3 < n) ? g_idata[i + 3] : 0.0f) + threadScanT.z;
    res = threadScanT.w;
  }

  res = scanLocalMem(res, s_data);
  __syncthreads();

  if (storeSum && tid == blockDim.x - 1) {
    g_blockSums[blockIdx.x] = res + threadScanT.w;
  }

  uint4 *outData = (uint4 *)g_odata;

  tempData.x = res;
  tempData.y = res + threadScanT.x;
  tempData.z = res + threadScanT.y;
  tempData.w = res + threadScanT.z;

  if (fullBlock || i + 3 < n) {
    outData[gid] = tempData;
  } else {
    if (i < n) {
      g_odata[i] = tempData.x;
      if ((i + 1) < n) {
        g_odata[i + 1] = tempData.y;
        if ((i + 2) < n) {
          g_odata[i + 2] = tempData.z;
        }
      }
    }
  }
}

__global__ void vectorAddUniform4(uint *d_vector, const uint *d_uniforms, const int n) {
  __shared__ uint uni[1];

  if (threadIdx.x == 0) {
    uni[0] = d_uniforms[blockIdx.x];
  }

  unsigned int address = threadIdx.x + (blockIdx.x * blockDim.x * 4);

  __syncthreads();

  for (int i = 0; i < 4 && address < n; i++) {
    d_vector[address] += uni[0];
    address += blockDim.x;
  }
}

// =================================================================================
// BASELINER IMPLEMENTATION
// =================================================================================

void SortKernel::setup(std::shared_ptr<cudaStream_t> stream) {
  int size = get_input()->m_size;
  long long bytes = size * sizeof(unsigned int);

  // 1. Allocate main buffers
  CHECK_CUDA(cudaMalloc((void **)&m_d_keys, bytes));
  CHECK_CUDA(cudaMalloc((void **)&m_d_vals, bytes));
  CHECK_CUDA(cudaMalloc((void **)&m_d_tempKeys, bytes));
  CHECK_CUDA(cudaMalloc((void **)&m_d_tempVals, bytes));

  // 2. Logic for Scan Block Sums Allocation
  uint maxNumScanElements = size;
  uint numScanElts = maxNumScanElements;
  uint level = 0;

  // Count levels
  do {
    uint numBlocks = std::max(1, (int)ceil((float)numScanElts / (4 * SCAN_BLOCK_SIZE)));
    if (numBlocks > 1) {
      level++;
    }
    numScanElts = numBlocks;
  } while (numScanElts > 1);

  m_numLevelsAllocated = level + 1;

  // Allocate Host array for pointers to device memory
  m_scanBlockSums = (unsigned int **)malloc((m_numLevelsAllocated) * sizeof(unsigned int *));

  numScanElts = maxNumScanElements;
  level = 0;

  // Allocate Device memory for each level
  do {
    uint numBlocks = std::max(1, (int)ceil((float)numScanElts / (4 * SCAN_BLOCK_SIZE)));
    if (numBlocks > 1) {
      CHECK_CUDA(cudaMalloc((void **)&(m_scanBlockSums[level]), numBlocks * sizeof(unsigned int)));
      level++;
    }
    numScanElts = numBlocks;
  } while (numScanElts > 1);

  CHECK_CUDA(cudaMalloc((void **)&(m_scanBlockSums[level]), sizeof(unsigned int)));

  // 3. Allocate Radix sort aux buffers
  // Each thread in the sort kernel handles 4 elements
  size_t numSortGroups = size / (4 * SORT_BLOCK_SIZE);

  CHECK_CUDA(cudaMalloc((void **)&m_d_counters, WARP_SIZE * numSortGroups * sizeof(unsigned int)));
  CHECK_CUDA(cudaMalloc((void **)&m_d_counterSums, WARP_SIZE * numSortGroups * sizeof(unsigned int)));
  CHECK_CUDA(cudaMalloc((void **)&m_d_blockOffsets, WARP_SIZE * numSortGroups * sizeof(unsigned int)));

  // 4. Initial Reset (Copy data)
  reset_kernel(stream);
}

void SortKernel::reset_kernel(std::shared_ptr<cudaStream_t> stream) {
  // Copy inputs to GPU
  long long bytes = get_input()->m_size * sizeof(unsigned int);
  CHECK_CUDA(cudaMemcpy(m_d_keys, get_input()->m_keys_host.data(), bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(m_d_vals, get_input()->m_vals_host.data(), bytes, cudaMemcpyHostToDevice));
}

void SortKernel::scanArrayRecursive(unsigned int *outArray, unsigned int *inArray, int numElements, int level,
                                    unsigned int **blockSums, std::shared_ptr<cudaStream_t> &stream) {
  // Kernels handle 8 elems per thread
  unsigned int numBlocks = std::max(1u, (unsigned int)ceil((float)numElements / (4.f * SCAN_BLOCK_SIZE)));
  unsigned int sharedEltsPerBlock = SCAN_BLOCK_SIZE * 2;
  unsigned int sharedMemSize = sizeof(uint) * sharedEltsPerBlock;

  bool fullBlock = (numElements == numBlocks * 4 * SCAN_BLOCK_SIZE);

  dim3 grid(numBlocks, 1, 1);
  dim3 threads(SCAN_BLOCK_SIZE, 1, 1);

  // execute the scan
  if (numBlocks > 1) {
    scan<<<grid, threads, sharedMemSize, *stream>>>(outArray, inArray, blockSums[level], numElements, fullBlock, true);
  } else {
    scan<<<grid, threads, sharedMemSize, *stream>>>(outArray, inArray, blockSums[level], numElements, fullBlock, false);
  }
  CHECK_CUDA(cudaGetLastError());

  if (numBlocks > 1) {
    scanArrayRecursive(blockSums[level], blockSums[level], numBlocks, level + 1, blockSums, stream);
    vectorAddUniform4<<<grid, threads, 0, *stream>>>(outArray, blockSums[level], numElements);
    CHECK_CUDA(cudaGetLastError());
  }
}

void SortKernel::run(std::shared_ptr<cudaStream_t> stream) {
  int numElements = get_input()->m_size;

  // Threads handle either 4 or two elements each
  const size_t radixGlobalWorkSize = numElements / 4;
  const size_t findGlobalWorkSize = numElements / 2;
  const size_t reorderGlobalWorkSize = numElements / 2;

  const size_t radixBlocks = radixGlobalWorkSize / SORT_BLOCK_SIZE;
  const size_t findBlocks = findGlobalWorkSize / SCAN_BLOCK_SIZE;
  const size_t reorderBlocks = reorderGlobalWorkSize / SCAN_BLOCK_SIZE;

  // Loop over the bits (Radix Sort)
  for (int startbit = 0; startbit < SORT_BITS; startbit += 4) {
    // 1. Sort Blocks
    radixSortBlocks<<<radixBlocks, SORT_BLOCK_SIZE, 4 * sizeof(uint) * SORT_BLOCK_SIZE, *stream>>>(
        4, startbit, (uint4 *)m_d_tempKeys, (uint4 *)m_d_tempVals, (uint4 *)m_d_keys, (uint4 *)m_d_vals);
    CHECK_CUDA(cudaGetLastError());

    // 2. Find Offsets
    findRadixOffsets<<<findBlocks, SCAN_BLOCK_SIZE, 2 * SCAN_BLOCK_SIZE * sizeof(uint), *stream>>>(
        (uint2 *)m_d_tempKeys, m_d_counters, m_d_blockOffsets, startbit, numElements, findBlocks);
    CHECK_CUDA(cudaGetLastError());

    // 3. Scan Recursive
    scanArrayRecursive(m_d_counterSums, m_d_counters, 16 * reorderBlocks, 0, m_scanBlockSums, stream);

    // 4. Reorder Data
    reorderData<<<reorderBlocks, SCAN_BLOCK_SIZE, 0, *stream>>>(
        startbit, (uint *)m_d_keys, (uint *)m_d_vals, (uint2 *)m_d_tempKeys, (uint2 *)m_d_tempVals, m_d_blockOffsets,
        m_d_counterSums, m_d_counters, reorderBlocks);
    CHECK_CUDA(cudaGetLastError());
  }
}

void SortKernel::teardown(std::shared_ptr<cudaStream_t> stream, SortOutput &output) {
  long long bytes = get_input()->m_size * sizeof(unsigned int);
  CHECK_CUDA(cudaMemcpy(output.m_keys_host.data(), m_d_keys, bytes, cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(output.m_vals_host.data(), m_d_vals, bytes, cudaMemcpyDeviceToHost));

  // Cleanup
  for (int i = 0; i < m_numLevelsAllocated; i++) {
    CHECK_CUDA(cudaFree(m_scanBlockSums[i]));
  }
  free(m_scanBlockSums); // Host array

  CHECK_CUDA(cudaFree(m_d_keys));
  CHECK_CUDA(cudaFree(m_d_vals));
  CHECK_CUDA(cudaFree(m_d_tempKeys));
  CHECK_CUDA(cudaFree(m_d_tempVals));
  CHECK_CUDA(cudaFree(m_d_counters));
  CHECK_CUDA(cudaFree(m_d_counterSums));
  CHECK_CUDA(cudaFree(m_d_blockOffsets));
}