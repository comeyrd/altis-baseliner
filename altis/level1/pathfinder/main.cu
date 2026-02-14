#include "PathFinder.hpp"
#include <baseliner/Axe.hpp>
#include <baseliner/Benchmark.hpp>
#include <baseliner/StoppingCriterion.hpp>
#include <baseliner/Suite.hpp>
#include <baseliner/Task.hpp>
namespace Path {
  static auto benchmark1 = Baseliner::CudaBenchmark()
                               .set_kernel<PathfinderKernel>()
                               .add_stat<Baseliner::Stats::Q1>()
                               .add_stat<Baseliner::Stats::Q3>()
                               .add_stat<Baseliner::Stats::Median>()
                               .add_stat<Baseliner::Stats::WithoutOutliers>()
                               .add_stat<Baseliner::Stats::MedianAbsoluteDeviation>();

  Baseliner::Axe axe = {"StoppingCriterion", "max_nb_repetition", {"100", "250", "500", "1000", "2000"}};

  Baseliner::SingleAxeSuite suite(std::make_shared<CudaBenchmark>(std::move(benchmark1)), std::move(axe));
  BASELINER_REGISTER_TASK(&suite);
} // namespace Path