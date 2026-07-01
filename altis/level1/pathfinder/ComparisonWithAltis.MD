# Validating PathFinder : Baseliner vs Altis
Porting PathFinder from Altis gave back different performance numbers, we will explain the difference in this document.

## What is PathFinder

PathFinder is a dynamic programming benchmark. It computes the minimum-weight path through a 2D grid (`rows x cols`), where each cell has a random "weight" (0-9), and the path can only move down one row at a time, stepping to one of the 3 cells directly below it (left, straight down, right).

For each row, every cell's result depends on 3 cells from the row above it (a halo of 1 column on each side). Because of this row-to-row dependency, the whole grid can't be computed in a single pass without synchronizing across the entire row before moving to the next one — and CUDA has no global synchronization primitive across all blocks within a single kernel launch.

To get around this, PathFinder uses "pyramid blocking" : each kernel launch computes `pyramid_height` rows at once, using shared memory within a block to handle the row-to-row dependency internally (with `__syncthreads()` between each row). Once `pyramid_height` rows are computed, the host launches another kernel for the next `pyramid_height` rows, using the previous kernel's output as input. This is why PathFinder launches many small kernels in sequence instead of one big one — `pyramid_height` trades off kernel launch overhead (smaller = more launches) against redundant halo recomputation at the edges of each block (larger = more wasted work per launch).

## How does PathFinder Altis runs kernels :

Per pass, the grid is split column-wise into blocks of `BLOCK_SIZE` (512) threads. Each block computes a "small block" of `smallBlockCol = BLOCK_SIZE - pyramid_height * HALO * 2` columns, with the remaining `pyramid_height * HALO` columns on each side used as halo/border to correctly propagate values without needing inter-block communication. `blockCols` (the grid dimension) is then `cols / smallBlockCol`, rounded up.

The full grid is computed by launching `ceil((rows - 1) / pyramid_height)` kernels in sequence, each one stepping `pyramid_height` rows further down the grid and using the previous kernel's output buffer as its input (ping-ponging between 2 result buffers).

It runs 32 instances : in `calc_path`, this whole sequence of kernel launches (the full grid, top to bottom) is repeated `numStreams` (=`instances`, default 32) times back to back, on the same buffers. This isn't concurrent/parallel work (unless `hyperq` is enabled, in which case each instance gets its own CUDA stream) — by default it's the same computation rerun sequentially 32 times, purely to get a larger, more stable sample for timing.

## How PathFinder Altis measures kernels : 
The measuring loop for altis is as follows : 

```cpp
cudaEventRecord(start, 0);
        dynproc_kernel<<<dimGrid, dimBlock>>>(
            MIN(pyramid_height, rows - t - 1), gpuWall, gpuResult[src],
            gpuResult[dst], cols, rows, t, borderCols);
        cudaDeviceSynchronize();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        CHECK_CUDA_ERROR();
        cudaEventElapsedTime(&elapsedTime, start, stop);
        kernelTime += elapsedTime * 1.e-3;
```

```bash
test                           atts            units       median         mean           stddev         min            max            
pathfinder_kernel_time         8192x8192       sec         0.701244       1.17024        1.157          0.692298       4.55605        
```

First to compare the results with baseliner, we need to divide the pathfinder_kernel_time by the number of instances ran by altis. Here it was 32. So 0.7s / 32 = 21,875ms.

There is a main issue in this measuring loop, the `cudaDeviceSynchronize()` will synchronize the host and the device, but recording with cuda_events records direclty on the device, so to each kernel measurement is added the synchronization time with the device, giving back huge measurements : 



If we remove the `cudaDeviceSynchronize()` and recompile, we get the following results : 
```bash
test                           atts            units       median         mean           stddev         min            max            
pathfinder_kernel_time         8192x8192       sec         0.423734       0.424373       0.00120858     0.423025       0.426766       
```

So a drop of 300ms per groups of kernel launches, and a much more stable distribution, with the stddev dropping, and the min and the max being much more close
With a individual job duration of =  0.423734s / 32 = 13,24ms

But for each 'job' multiple kernels would be launched, each with a `cudaEventSynchronize(stop);`.
Not enqueing the kernels and measuring the times after all the launches adds device synchronization and scheduling to the measurement.
Because individual kernel times are not used, we can measure the whole loop and report that.
Moving the measurement outside the kernel launch loop.

```cpp
for (int s = 0; s < numStreams; s++) {
    if (hyperq) {
      cudaEventRecord(start, streams[s]);
    } else {
      cudaEventRecord(start, 0);
    }
    for (int t = 0; t < rows - 1; t += pyramid_height) {
      int temp = src;
      src = dst;
      dst = temp;

      if (hyperq) {
        dynproc_kernel<<<dimGrid, dimBlock, 0, streams[s]>>>(
            MIN(pyramid_height, rows - t - 1), gpuWall, gpuResult[src],
            gpuResult[dst], cols, rows, t, borderCols);
      } else {
        dynproc_kernel<<<dimGrid, dimBlock>>>(
            MIN(pyramid_height, rows - t - 1), gpuWall, gpuResult[src],
            gpuResult[dst], cols, rows, t, borderCols);
      }
    }
    if (hyperq) {
      cudaEventRecord(stop, streams[s]);
    } else {
      cudaEventRecord(stop, 0);
    }
    cudaEventElapsedTime(&elapsedTime, start, stop);
    kernelTime += elapsedTime * 1.e-3;
  }
```
The results are the following : 
```bash
test                           atts            units       median         mean           stddev         min            max            
pathfinder_kernel_time         8192x8192       sec         0.319932       0.319927       7.40398e-05    0.319794       0.320073       
```
Now with the final updated version of altis' pathfinder, with an individual instance duration of = 0.319932s / 32 = 9,99ms

Here we have a std dev that is even smaller. 
With a few updates we drastically changed the measurements and results of this workload.

## How Baseliner Altis measures kernels : 

Baseliner runs just one instance of the workload by batch of 20, and with a total of 500 workload runs.
We ran it using the same setup (same random number generation and same parameters, using the pathfinder-protocol.json found in this directory)
with the following results : 
```bash
./altis_exec run -pf ../../altis/level1/pathfinder/pathfinder-protocol.json 
Runnning protocol : ../../altis/level1/pathfinder/pathfinder-protocol.json...
Running Campaign default, Recipe : default
Workload : Pathfinder (default)
Backend : cuda (default)
Warning : Terminal output is truncated due to terminal size missing harmonic_mean (ms), warmup_time (ms), host_setup_time (ms), cpu_time (ms), device_setup_time (ms), fetch_results_time (ms), memory_usage (bytes) | full output in report file
repetitions ()|batch_count ()|median (ms) |mean (ms)   |coefficient_of_variation (%)|memory_bandwidth (GB/s)|harmonic_mean (ms)|
--------------+--------------+------------+------------+----------------------------+-----------------------+------------------+
500           |229           |9.967616    |9.968960    |0.054341                    |26.933886              |9.968892          |
Report saved to result-06n2t30g7b4ua.json
```
With a median of 9.967616 ms, baseliner's version is .3% quicker, this is attributed to baseliner's measurement loop that uses a blocking kernel to enqueue kernels.

The results are considered similar.

