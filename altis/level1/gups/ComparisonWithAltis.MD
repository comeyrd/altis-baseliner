# Validating GUPS : Baseliner vs Altis

## What is GUPS
GUPS (Giga UPdates per Second) is a benchmark that measures how fast a system's memory can handle completely random read-modify-write operations. It computes these updates across a massive 1D array (or table) of 64-bit numbers.

Because the memory addresses are generated at random (using a mathematical formula), the GPU cannot rely on its fast memory caches. Every access is forced to go straight to global memory. Furthermore, because thousands of threads are picking random addresses at the same time, multiple threads will inevitably try to update the exact same memory locations simultaneously, creating massive collisions.

To prevent data corruption during these collisions, GUPS relies on hardware atomic operations to safely lock and update memory. To allow massive parallelization, it uses a mathematical "jump-ahead" function so every thread can instantly calculate its own starting random number without waiting for the threads before it.

## How does GUPS Altis runs kernels :
Per pass, Altis uses a highly tailored kernel setup: the block size is hardcoded to exactly one warp (warpSize = 32 threads), and the grid size is calculated to perfectly fill the GPU's multiprocessors. This maximizes occupancy and essentially treats the GPU as if it is scheduling individual warps rather than larger thread blocks.

The total workload across the grid is exactly 4 updates per array element. This total workload is divided evenly among all the launched threads. Each thread calculates its specific start and end boundaries, seeds its own random number generator, and loops through its assigned chunk of updates.

The full execution is split into two distinct testing phases: one using atomicCAS (Compare-And-Swap) and another using native atomicXor. Each phase is repeated for a set number of passes. For every single pass, the host launches a sequence of three kernels: an init kernel to reset the table, the main bench kernel to perform the timed updates, and a check kernel to verify that no updates were dropped.

## How GUPS Altis measures kernels : 

The measuring loop is as follows : 

```cpp
for (int i = 0; i < passes; i++) {
    d_init<<<grid, thread>>>(n, d_t);
    checkCudaErrors(cudaEventRecord(begin));
    d_bench<ATOMICTYPE_CAS><<<grid, thread>>>(n, d_t);
    checkCudaErrors(cudaEventRecord(end));
    checkCudaErrors(cudaEventSynchronize(end));

    float ms;
    checkCudaErrors(cudaEventElapsedTime(&ms, begin, end));

    double time = ms * 1.0e-3;
    DB.AddResult("Elapsed time", atts, "seconds", time);
    double gups = 4 * n / (double) ms * 1.0e-6;
    DB.AddResult("Giga Updates per second", atts, "GUP/s", gups);

    cudaMemset(d_error, 0, sizeof(uint32_t));
    d_check<<<grid, thread>>>(n, d_t);
    uint32_t h_error;
    checkCudaErrors(cudaMemcpy(&h_error, p_error, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    if (op.getOptionBool("verbose")) {
      printf("Verification (ATOMICTYPE_CAS): Found %u errors.\n", h_error);
    }
  }
```

and the results with the default : 
```bash
Chose device: name='Tesla V100-DGXS-32GB' index=0
Running GUPS
Total table size = 1048576 (8388608 bytes.)
Using GPU 0 of 4 GPUs.
Warp size = 32.
Multi-processor count = 80.
Max threads per multi-processor = 2048.

test                       atts                 units         median         mean           stddev         min            max            
Elapsedtime                ATOMICTYPE_CAS       seconds       0.00269158     0.00269138     8.14134e-06    0.00267776     0.00270643     
Elapsedtime                ATOMICTYPE_XOR       seconds       0.00089344     0.000895078    1.20158e-05    0.000883712    0.000928768    
GigaUpdatespersecond       ATOMICTYPE_CAS       GUP/s         1.5583         1.55844        0.00471089     1.54975        1.56635        
GigaUpdatespersecond       ATOMICTYPE_XOR       GUP/s         4.69456        4.68678        0.0612527      4.51599        4.74623        
```
The median is 2.69ms per kernel launch for ATOMICTYPE_CAS and 0.89ms per kernel launch for ATOMICTYPE_XOR.

If we compare with baseliner, we get the following results : 

```bash
Running Campaign cas, Recipe : default
Workload : GupsXOR (default)
Backend : cuda (default)
Warp size 32
Multi-processor count 80
Max threads per multi-processor 2048
Warning : Terminal output is truncated due to terminal size missing memory_bandwidth (GB/s), harmonic_mean (ms), arithmetic_bandwidth (GFLOP/S), arithmetic_intensity (), warmup_time (ms), host_setup_time (ms), cpu_time (ms), device_setup_time (ms), fetch_results_time (ms), memory_usage (bytes), arithmetic_usage (Flops) | full output in report file
repetitions ()|batch_count ()|median (ms) |mean (ms)   |coefficient_of_variation (%)|memory_bandwidth (GB/s)|
--------------+--------------+------------+------------+----------------------------+-----------------------+
506           |38            |0.897024    |0.898947    |0.148501                    |9.337955               |
Workload : GupsCAS (default)
Backend : cuda (default)
Warp size 32
Multi-processor count 80
Max threads per multi-processor 2048
500           |68            |2.744320    |2.744003    |0.144110                    |3.057119               |
Report saved to result-06n303ouhr2o5.json
```
We see here that baseliner's version is slower on a significant size ( 0.03ms ) and i am not able to showcase why