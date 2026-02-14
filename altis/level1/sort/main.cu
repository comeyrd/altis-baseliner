#include "SortKernel.hpp"
#include <baseliner/Benchmark.hpp>
#include <baseliner/StoppingCriterion.hpp>
#include <baseliner/Task.hpp>

namespace Sort {
  static auto benchmark1 = Baseliner::CudaBenchmark()
                               .set_kernel<SortKernel>()
                               .set_stopping_criterion<Baseliner::ConfidenceIntervalMedianSC>()
                               .add_stat<Baseliner::Stats::Q1>()
                               .add_stat<Baseliner::Stats::Q3>()
                               .add_stat<Baseliner::Stats::Median>()
                               .add_stat<Baseliner::Stats::WithoutOutliers>()
                               .add_stat<Baseliner::Stats::MedianAbsoluteDeviation>();
  BASELINER_REGISTER_TASK(&benchmark1);

} // namespace Sort