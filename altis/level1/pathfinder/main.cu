#include "PathFinder.hpp"
#include <baseliner/Axe.hpp>
#include <baseliner/Benchmark.hpp>
#include <baseliner/Executable.hpp>
#include <baseliner/Runner.hpp>
#include <baseliner/StoppingCriterion.hpp>
namespace Path {

  auto runner = std::make_shared<Baseliner::Runner<Baseliner::PathfinderKernel, Baseliner::Backend::CudaBackend>>();

  Baseliner::Axe axe = {"StoppingCriterion", "max_nb_repetition", {"100", "250", "500", "1000", "2000"}};
  Baseliner::SingleAxeBenchmark bench(runner, axe);
  BASELINER_REGISTER_EXECUTABLE(&bench);
} // namespace Path