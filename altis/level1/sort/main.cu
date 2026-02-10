#include "SortKernel.hpp"
#include <baseliner/Executable.hpp>
#include <baseliner/Runner.hpp>
#include <baseliner/StoppingCriterion.hpp>

namespace Sort {
  auto stop = std::make_unique<Baseliner::ConfidenceIntervalMedianSC>();
  Baseliner::Runner<Baseliner::SortKernel, Baseliner::Backend::CudaBackend> runner_act(std::move(stop));
  BASELINER_REGISTER_EXECUTABLE(&runner_act);

} // namespace Sort