#ifndef PATHFINDER_WORKLOAD_HPP
#define PATHFINDER_WORKLOAD_HPP

#include <baseliner/core/Workload.hpp>

#include <algorithm>
#include <cmath>
#include <random>
#include <string>
#include <vector>

constexpr int PATHFINDER_BLOCK_SIZE = 512;
constexpr int PATHFINDER_HALO = 1;

template <typename BackendT>
class PathfinderWorkload : public Baseliner::IWorkload<BackendT> {
public:
  using backend = typename PathfinderWorkload::backend;

  PathfinderWorkload() = default;

  auto algo() -> std::string override {
    return "Pathfinder";
  }

  auto number_of_bytes() -> std::optional<size_t> override {
    return sizeof(int) * m_rows * m_cols + sizeof(int) * m_cols * 2;
  }

  // Host setup
  void setup_host_random_generated() override {
    // Pathfinder is memory bound (3 flops per int read), so 1 unit of
    // work size == 1 MiB of grid data, not 32 MFLOP.
    //
    // total bytes ≈ rows * cols * sizeof(int)
    //
    // m_base_rows/m_base_cols define the *shape* of the work_size==1 case
    // (default 512x512 -> exactly 1 MiB). To scale total bytes linearly
    // with work_size while preserving that aspect ratio, each dimension
    // is scaled by sqrt(work_size), since rows*cols grows with the
    // square of the per-dimension scale factor.
    const double scale = std::sqrt(static_cast<double>(this->get_work_size()));

    m_pyramid_height = m_base_pyramid_height;

    m_rows = std::max(1, static_cast<int>(std::llround(m_base_rows * scale)));
    m_cols = std::max(1, static_cast<int>(std::llround(m_base_cols * scale)));

    // Guard against degenerate shapes at very small work sizes / large
    // pyramid heights: cols must be large enough for the halo on both sides.
    const int min_cols = m_pyramid_height * PATHFINDER_HALO * 2 + 1;
    if (m_cols < min_cols) {
      m_cols = min_cols;
    }

    m_h_data.resize(static_cast<size_t>(m_rows) * m_cols);
    m_h_result.resize(m_cols);
    srand(this->get_seed());
    for (int i = 0; i < m_rows * m_cols; i++) {
      m_h_data[i] = rand() % 10;
    }

    m_borderCols = m_pyramid_height * PATHFINDER_HALO;
    m_smallBlockCol = PATHFINDER_BLOCK_SIZE - m_pyramid_height * PATHFINDER_HALO * 2;
    m_blockCols = m_cols / m_smallBlockCol + ((m_cols % m_smallBlockCol == 0) ? 0 : 1);
  }

  void setup_host_from_file(std::string &path) override {
    setup_host_random_generated();
  }

  void inner_save_setup(std::string &path) override {};

  void setup_device(typename backend::stream_t stream) override;
  void reset_device(typename backend::stream_t stream) override;
  auto run(typename backend::stream_t stream) -> typename backend::launch_result_t override;
  void fetch_results(typename backend::stream_t stream) override;
  void free() override {
    m_h_data.clear();
    m_h_result.clear();
  }

protected:
  void register_options() override {
    Baseliner::IWorkload<BackendT>::register_options();
    this->add_option("PathfinderWorkload", "rows", "Base number of rows at work_size=1", m_base_rows);
    this->add_option("PathfinderWorkload", "cols", "Base number of columns at work_size=1", m_base_cols);
    this->add_option("PathfinderWorkload", "pyramid_height", "Pyramid height", m_base_pyramid_height);
  }

private:
  // Default shape gives rows*cols*sizeof(int) == 1 MiB at work_size == 1
  // (512 * 512 * 4 bytes = 1,048,576 bytes).
  int m_base_rows = PATHFINDER_BLOCK_SIZE;
  int m_base_cols = PATHFINDER_BLOCK_SIZE;
  int m_base_pyramid_height = 4;

  int m_rows = 0;
  int m_cols = 0;
  int m_pyramid_height = 0;

  int m_borderCols = 0;
  int m_smallBlockCol = 0;
  int m_blockCols = 0;

  std::vector<int> m_h_data;
  std::vector<int> m_h_result;

  int *m_d_gpuWall = nullptr;
  int *m_d_gpuResult[2] = {nullptr, nullptr};

  int m_final_ret = 0;
};

#endif // PATHFINDER_WORKLOAD_HPP