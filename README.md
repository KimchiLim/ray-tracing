# ray-tracing

Ray tracing project for UChicago MPCS 51087 High Performance Computing.

Multiple implementations of a Monte Carlo ray tracing simulation using varying methods of parallelization: pure, serial C (for comparison); CPU-multithreading with OpenMP; single GPU acceleration using CUDA; and GPU acceleration with multiple devices using OpenMPI and CUDA.

# Usage

All executables can be generated using the Makefile and run like so:

```
make ray_tracing_<serial/omp/cuda/mpi>
./ray_tracing_<serial/omp/cuda/mpi> <number of generated rays> <image resolution> <additional inputs>
```

The OpenMP implementation additionally requires an input for the number of CPU threads, and the CUDA and MPI implementations require inputs for the number of device blocks and GPU threads per block. The Makefile is currently set to compile for the NVIDIA Tesla V100 --- change the arch flag accordingly for different hardware.

Running the executable will generate a binary file `sphere.bin` with the image data. The full image can be generated with `python3 plot.py`.