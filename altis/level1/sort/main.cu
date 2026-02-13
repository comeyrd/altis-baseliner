#include "SortKernel.hpp"
#include <baseliner/Executable.hpp>
#include <baseliner/Runner.hpp>
#include <baseliner/StoppingCriterion.hpp>

namespace Sort {
  auto runner_act = Baseliner::Runner<Baseliner::SortKernel, Baseliner::Backend::CudaBackend>()
                        .set_stopping_criterion<Baseliner::ConfidenceIntervalMedianSC>()
                        .add_stat<Baseliner::Stats::Q1>()
                        .add_stat<Baseliner::Stats::Q3>()
                        .add_stat<Baseliner::Stats::Median>()
                        .add_stat<Baseliner::Stats::WithoutOutliers>()
                        .add_stat<Baseliner::Stats::MedianAbsoluteDeviation>();
  BASELINER_REGISTER_EXECUTABLE(&runner_act);

} // namespace Sort