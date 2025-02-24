CC=gcc
CFLAGS=-I. -lm -O3 -fopenmp

ray_tracing_serial: ray_tracing_serial.c pcg/pcg_basic.c
	$(CC) -o ray_tracing_serial ray_tracing_serial.c pcg/pcg_basic.c $(CFLAGS)

ray_tracing_omp: ray_tracing_omp.c pcg/pcg_basic.c
	$(CC) -o ray_tracing_omp ray_tracing_omp.c pcg/pcg_basic.c $(CFLAGS)

ray_tracing_cuda: ray_tracing_cuda.cu
	nvcc -arch=compute_70 -o ray_tracing_cuda_sp ray_tracing_cuda.cu -lm -O3

ray_tracing_mpi: ray_tracing_mpi.cu
	nvcc -arch=compute_70 -I$(mpicxx --showme:incdirs) -L$(mpicxx --showme:libdirs) -lmpi -lmpi_cxx ray_tracing_mpi.cu -o ray_tracing_mpi

clean:
	rm ray_tracing_serial ray_tracing_omp ray_tracing_cuda ray_tracing_mpi ray_tracing_cuda_dp *.bin *.png *.out *.err