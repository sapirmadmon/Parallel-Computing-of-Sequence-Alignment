all:  
	mpicxx -fopenmp -c final_project.c -o final_project.o
	nvcc -I../inc -c cudaFunctions.cu -o cudaFunctions.o
	mpicxx -fopenmp -o finalexec final_project.o cudaFunctions.o /usr/local/cuda/lib64/libcudart_static.a -ldl -lrt

run:  
	mpiexec -np 2 ./finalexec input.txt
	
runOn2:
	mpiexec -np 2 -machinefile  mf  -map-by  node  ./finalexec input.txt
	              
clean:
	rm -f *.o finalexec
