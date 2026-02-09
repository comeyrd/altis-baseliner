#include "PathFinder.hpp"
#include <baseliner/Executable.hpp>
#include <baseliner/Runner.hpp>
#include <baseliner/StoppingCriterion.hpp>

auto stop = Baseliner::ConfidenceIntervalMedianSC();
Baseliner::Runner<Baseliner::PathfinderKernel, Baseliner::Backend::CudaBackend> runner_act(stop);
BASELINER_REGISTER_EXECUTABLE(&runner_act);