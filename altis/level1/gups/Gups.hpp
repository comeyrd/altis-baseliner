#ifndef GUPS_WORKLOAD_HPP
#define GUPS_WORKLOAD_HPP

#include <baseliner/core/Workload.hpp>

#include <cstdint>
#include <string>

template <typename BackendT>
class GupsWorkload : public Baseliner::IWorkload<BackendT> {
public:
  using backend = typename GupsWorkload::backend;

  GupsWorkload() = default;

  // Identification
  auto algo() -> std::string override {
    return "Gups";
  }

  // Metrics
  auto number_of_bytes() -> std::optional<size_t> override {
    return m_n * sizeof(uint64_t);
  }

  auto number_of_floating_point_operations() -> std::optional<size_t> override {
    return 4 * m_n;
  }

  // Host setup
  void setup_host_random_generated() override {
    compute_size();
    starts();
  }

  void setup_host_from_file(std::string &path) override {
    compute_size();
    starts();
  }

  void inner_save_setup(std::string &path) override {};

  void setup_device(typename backend::stream_t stream) override;
  void reset_device(typename backend::stream_t stream) override;
  void fetch_results(typename backend::stream_t stream) override;
  void free() override {
    m_h_error = 0;
  }

  // Validation
  auto validate() -> bool override {
    return m_h_error == 0;
  }

protected:
  void register_options() override {
    Baseliner::IWorkload<BackendT>::register_options();
    this->add_option("GupsWorkload", "base_n", "Table size (number of elements) at work_size=1", m_base_n);
  }
  void compute_size() {
    size_t target = m_base_n * this->get_work_size();
    m_n = 1;
    while (m_n < target) {
      m_n <<= 1;
    }
  }

  void starts() {
    uint64_t temp = 1;
    for (ptrdiff_t i = 0; i < 64; i++) {
      m_m2[i] = temp;
      temp = (temp << 1) ^ ((int64_t)temp < 0 ? POLY : 0);
      temp = (temp << 1) ^ ((int64_t)temp < 0 ? POLY : 0);
    }
  }

  static constexpr uint64_t POLY = 0x0000000000000007ULL;

  size_t m_base_n = 16384;
  size_t m_n = 0;

  uint64_t m_m2[64];

  uint32_t m_h_error = 0;

  void *m_d_t = nullptr;

  int m_grid_size = 0;
  int m_thread_size = 0;
};

#endif // GUPS_WORKLOAD_HPP