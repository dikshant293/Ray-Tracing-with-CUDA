compile:
	nvcc ray_tracing.cu -use_fast_math -Xcompiler -fopenmp -O3 -o ray_tracing -lmpi -arch=sm_61

clean:
	\rm -f *.o a.out ray_tracing.exe ray_tracing ray_tracing  matrix.dat animation.dat startend.dat*~ *# 

test_serial: compile
	time ./ray_tracing 1000000000 1000 10000 1024 1 1
	./plt.sh omp-image.dat serial-1000
	time ./ray_tracing 1000000000 100 10000 1024 1 1
	./plt.sh omp-image.dat serial-100

test_omp: compile
	time ./ray_tracing 1000000000 1000 10000 1024 1
	./plt.sh omp-image.dat omp-1000
	time ./ray_tracing 1000000000 100 10000 1024 1
	./plt.sh omp-image.dat omp-

test_cuda: compile
	time ./ray_tracing 1000000000 1000 10000 1024 2
	./plt.sh cuda-image.dat [CUDA]
	time ./ray_tracing 1000000000 100 10000 1024 2
	./plt.sh cuda-image.dat [CUDA]

test_mpi: compile
	time mpirun -np 2 --oversubscribe ./ray_tracing 1000000000 1000 10000 1024 3
	./plt.sh cuda-image.dat MPI-1000
	time mpirun -np 2 --oversubscribe ./ray_tracing 1000000000 100 10000 1024 3
	./plt.sh cuda-image.dat MPI-100

test_cuda_v100: compile
	time ./ray_tracing 1000000000 1000 10000 1024 2
	./plt.sh cuda-image.dat v100-1000
	time ./ray_tracing 1000000000 100 10000 1024 2
	./plt.sh cuda-image.dat v100-100

test_cuda_rtx6000: compile
	time ./ray_tracing 1000000000 1000 10000 1024 2
	./plt.sh cuda-image.dat rtx6000-1000
	time ./ray_tracing 1000000000 100 10000 1024 2
	./plt.sh cuda-image.dat rtx6000-100

test_cuda_a100: compile
	time ./ray_tracing 1000000000 1000 10000 1024 2
	./plt.sh cuda-image.dat a100-1000
	time ./ray_tracing 1000000000 100 10000 1024 2
	./plt.sh cuda-image.dat a100-100
