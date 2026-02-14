#ifndef SORT_KERNEL_HPP
#define SORT_KERNEL_HPP

#include "cuda_runtime.h"
#include <algorithm>
#include <baseliner/Kernel.hpp>
#include <baseliner/Options.hpp>
#include <baseliner/backend/cuda/CudaBackend.hpp>
#include <iostream>
#include <random>
#include <vector>

using namespace Baseliner;

constexpr int DEFAULT_SIZE = 1024 * 1024;
#define SORT_BITS 32

class SortInput : public Baseliner::IInput {
public:
  void register_options() override {
    IInput::register_options();
    add_option("SortInput", "base_size", "The number of elements to sort", m_base_size);
  }

  void generate_random() override {
    std::default_random_engine gen(get_seed());
    std::uniform_int_distribution<unsigned int> dist(0, 1024);

    for (int i = 0; i < m_size; i++) {
      m_keys_host[i] = i % 1024; // Pattern from original benchmark
      m_vals_host[i] = dist(gen);
    }
  }

  explicit SortInput()
      : Baseliner::IInput() {
    allocate();
  }

  void allocate() override {
    m_size = m_base_size * get_work_size();
    m_keys_host.resize(m_size);
    m_vals_host.resize(m_size);
  }

  int m_base_size = DEFAULT_SIZE;
  int m_size;
  std::vector<unsigned int> m_keys_host;
  std::vector<unsigned int> m_vals_host;
};

class SortOutput : public Baseliner::IOutput<SortInput> {
public:
  explicit SortOutput(std::shared_ptr<const SortInput> input)
      : Baseliner::IOutput<SortInput>(input) {
    m_keys_host.resize(get_input()->m_size);
    m_vals_host.resize(get_input()->m_size);
  }

  std::vector<unsigned int> m_keys_host;
  std::vector<unsigned int> m_vals_host;

  friend std::ostream &operator<<(std::ostream &os, const SortOutput &out) {
    os << "Output (First 10): " << std::endl;
    int limit = std::min((int)out.get_input()->m_size, 10);
    for (int i = 0; i < limit; i++) {
      os << "K: " << out.m_keys_host[i] << " V: " << out.m_vals_host[i] << std::endl;
    }
    return os;
  }
  bool operator==(const SortOutput &other) const {
    if (get_input()->m_size != other.get_input()->m_size) {
      return false;
    }

    for (int i = 0; i < get_input()->m_size; i++) {
      if (m_keys_host[i] != other.m_keys_host[i]) {
        return false;
      }
      if (m_vals_host[i] != other.m_vals_host[i]) {
        return false;
      }
    }
    return true;
  }
};

class SortKernel : public Baseliner::ICudaKernel<SortInput, SortOutput> {
public:
  SortKernel(std::shared_ptr<const SortInput> input)
      : Baseliner::ICudaKernel<SortInput, SortOutput>(input) {
  }
  std::string name() override {
    return "SortKernel";
  }

  void setup(std::shared_ptr<cudaStream_t> stream) override;
  void reset_kernel(std::shared_ptr<cudaStream_t> stream) override;
  void run(std::shared_ptr<cudaStream_t> stream) override;
  void teardown(std::shared_ptr<cudaStream_t> stream, SortOutput &output) override;

private:
  // Helper function for the recursive scan step
  void scanArrayRecursive(unsigned int *outArray, unsigned int *inArray, int numElements, int level,
                          unsigned int **blockSums, std::shared_ptr<cudaStream_t> &stream);

  // Device pointers
  unsigned int *m_d_keys = nullptr;
  unsigned int *m_d_vals = nullptr;
  unsigned int *m_d_tempKeys = nullptr;
  unsigned int *m_d_tempVals = nullptr;

  // Radix counters
  unsigned int *m_d_counters = nullptr;
  unsigned int *m_d_counterSums = nullptr;
  unsigned int *m_d_blockOffsets = nullptr;

  // Scan block sums (Array of device pointers)
  unsigned int **m_scanBlockSums = nullptr;
  unsigned int m_numLevelsAllocated = 0;
};

#endif // SORT_KERNEL_HPP