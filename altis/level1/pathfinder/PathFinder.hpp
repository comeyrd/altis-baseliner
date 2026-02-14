#ifndef PATHFINDER_KERNEL_HPP
#define PATHFINDER_KERNEL_HPP

#include "cuda_runtime.h"
#include <algorithm>
#include <baseliner/Kernel.hpp>
#include <baseliner/Options.hpp>
#include <baseliner/backend/cuda/CudaBackend.hpp>
#include <iostream>
#include <vector>

using namespace Baseliner;
constexpr int DEFAULT_ROWS = 8 * 1024;
constexpr int DEFAULT_COLS = 8 * 1024;
constexpr int DEFAULT_PYRAMID_HEIGHT = 4;

class PathfinderInput : public Baseliner::IInput {
public:
  void register_options() override {
    IInput::register_options();
    add_option("Pathfinder", "rows", "Number of rows", m_rows);
    add_option("Pathfinder", "cols", "Number of columns", m_cols);
    add_option("Pathfinder", "pyramid_height", "Height of the pyramid", m_pyramid_height);
  }

  void allocate() override {
    // Apply work_size to rows to scale the workload
    m_rows = (m_rows <= 0 ? DEFAULT_ROWS : m_rows) * get_work_size();
    m_cols = (m_cols <= 0 ? DEFAULT_COLS : m_cols) * get_work_size();
    m_pyramid_height = (m_pyramid_height <= 0 ? DEFAULT_PYRAMID_HEIGHT : m_pyramid_height) * get_work_size();

    m_flat_data.resize(m_rows * m_cols);
  }

  void generate_random() override;

  explicit PathfinderInput()
      : Baseliner::IInput() {
    m_rows = DEFAULT_ROWS;
    m_cols = DEFAULT_COLS;
    m_pyramid_height = DEFAULT_PYRAMID_HEIGHT;
    allocate();
  }

  int m_rows;
  int m_cols;
  int m_pyramid_height;
  std::vector<int> m_flat_data;
};

class PathfinderOutput : public Baseliner::IOutput<PathfinderInput> {
public:
  explicit PathfinderOutput(std::shared_ptr<const PathfinderInput> input)
      : Baseliner::IOutput<PathfinderInput>(input) {
    m_result_host.resize(input->m_cols);
  }

  std::vector<int> m_result_host;

  bool operator==(const PathfinderOutput &other) const {
    if (get_input()->m_cols != other.get_input()->m_cols)
      return false;
    for (size_t i = 0; i < m_result_host.size(); i++) {
      if (m_result_host[i] != other.m_result_host[i]) {
        return false;
      }
    }
    return true;
  }

  friend std::ostream &operator<<(std::ostream &os, const PathfinderOutput &thing) {
    os << "Result (first 20 elements): ";
    size_t limit = std::min((size_t)20, thing.m_result_host.size());
    for (size_t i = 0; i < limit; i++) {
      os << thing.m_result_host[i] << ", ";
    }
    os << "..." << std::endl;
    return os;
  }
};

class PathfinderKernel : public Baseliner::ICudaKernel<PathfinderInput, PathfinderOutput> {
public:
  PathfinderKernel(std::shared_ptr<const PathfinderInput> input)
      : Baseliner::ICudaKernel<PathfinderInput, PathfinderOutput>(std::move(input)),
        m_d_gpuWall(nullptr),
        m_final_ret_idx(0) {
    m_d_gpuResult[0] = nullptr;
    m_d_gpuResult[1] = nullptr;
  }
  std::string name() override {
    return "PathfinderKernel";
  }

  void setup(std::shared_ptr<cudaStream_t> stream) override;

  void reset_kernel(std::shared_ptr<cudaStream_t> stream) override;

  void run(std::shared_ptr<cudaStream_t> stream) override;

  void teardown(std::shared_ptr<cudaStream_t> stream, PathfinderOutput &output) override;

private:
  int *m_d_gpuWall;
  int *m_d_gpuResult[2];
  int m_final_ret_idx; // Tracks which buffer has the final result
};

#endif // PATHFINDER_KERNEL_HPP