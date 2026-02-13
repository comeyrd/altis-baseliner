#include "PathFinder.hpp"
#include <baseliner/Axe.hpp>
#include <baseliner/Benchmark.hpp>
#include <baseliner/Executable.hpp>
#include <baseliner/Runner.hpp>
#include <baseliner/StoppingCriterion.hpp>
namespace Path {

  auto runner = Baseliner::Runner<Baseliner::PathfinderKernel, Baseliner::Backend::CudaBackend>()
                    .add_stat<Baseliner::Stats::Q1>()
                    .add_stat<Baseliner::Stats::Q3>()
                    .add_stat<Baseliner::Stats::Median>()
                    .add_stat<Baseliner::Stats::WithoutOutliers>()
                    .add_stat<Baseliner::Stats::MedianAbsoluteDeviation>();

  Baseliner::Axe axe = {"StoppingCriterion", "max_nb_repetition", {"100", "250", "500", "1000", "2000"}};
  Baseliner::SingleAxeBenchmark
      bench(std::make_shared<Baseliner::Runner<Baseliner::PathfinderKernel, Baseliner::Backend::CudaBackend>>(
                std::move(runner)),
            axe);
  BASELINER_REGISTER_EXECUTABLE(&bench);
} // namespace Path