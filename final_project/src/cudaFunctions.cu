//Sapir Madmon 209010230

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "cudaFunctions.h"


//the func called from host(CPU) and executed on device (GPU) 
__global__ void calcBestScoreViaCuda(double *d_matW, char *seq1, char *seq2, int offSet, int numSeq2, double *d_arrScore, int lenSeq2) 
{
	int index_seq1, index_seq2;
	double score = 0;
	int tid = blockDim.x * blockIdx.x + threadIdx.x; //each tid is index of mutant
	
	tid++;	//should be increased by 1 (because the mutant index starts at position 1)

	if (tid > lenSeq2)
		return;

	for (index_seq2 = 0; index_seq2 < lenSeq2; index_seq2++) 
	{
		index_seq1 = index_seq2 + offSet;
		if (index_seq2 >= tid)
			index_seq1++;
		
		score += d_matW[NUM_ABC * (seq2[index_seq2] - 'A') + (seq1[index_seq1] - 'A')];
	}
	
	tid--; 
	d_arrScore[tid] = score;
}


double* computeArrScoreOnGPU(double *d_matW, char *seq1, char *seq2, int offSet, int numSeq2, int seq2Len) 
{	
	double *d_arrScore;
	size_t size_seq2Len = seq2Len * sizeof(double);
	
	//Allocate memory on GPU to copy the data from the host
	cudaError(cudaMalloc((void**) &d_arrScore, size_seq2Len), "Failed to allocate device memory arrScore - %s\n");

	double *arrScore = (double*) calloc(seq2Len, sizeof(double));  //arr score for all results 
	
	int threadsPerBlock = 256;	//number of threads per block
	int blocksPerGrid = (seq2Len + threadsPerBlock - 1) / threadsPerBlock; //compute number of blocks per grid  

	//kernel calc the best score and create array of result  
	calcBestScoreViaCuda<<<blocksPerGrid, threadsPerBlock>>>(d_matW, seq1, seq2, offSet, numSeq2, d_arrScore, seq2Len);
	
	//Copy the result (arrScore) from GPU to the host memory.
	cudaMemcpy(arrScore, d_arrScore, size_seq2Len, cudaMemcpyDeviceToHost);
	cudaError(cudaFree(d_arrScore), "Failed to free device arrScore - %s\n");

	return arrScore;
}


//Copy data from host to the GPU memory
double* allocatedWeightMat(double matW[NUM_ABC][NUM_ABC]) 
{
	double *d_matW = NULL;
	size_t size_matW = NUM_ABC * NUM_ABC * sizeof(double);
	
	//Allocate memory on GPU to copy the data from the host
	cudaError(cudaMalloc((void**) &d_matW, size_matW), "Failed to copy WeightMat from host to device - %s\n");
	cudaMemcpy(d_matW, matW, size_matW, cudaMemcpyHostToDevice); //Copy matW from host to the GPU memory

	return d_matW;
}


//Copy data from host to the GPU memory
char* allocatedSeq1OnGPU(char *seq1)
{
	char *d_seq1 = NULL;
	size_t size_seq1 = (strlen(seq1)) * sizeof(char);
	
	//Allocate memory on GPU to copy the data from the host
	cudaError(cudaMalloc((void**) &d_seq1, size_seq1), "Failed to copy seq1 from host to device - %s\n");
	cudaMemcpy(d_seq1, seq1, size_seq1, cudaMemcpyHostToDevice); //Copy seq1 from host to the GPU memory

	return d_seq1;
}


//Copy data from host to the GPU memory
char* allocatedSeq2OnGPU(char *seq2) 
{
	char *d_seq2 = NULL;
	size_t size_seq2 = (strlen(seq2)) * sizeof(char);
	
	//Allocate memory on GPU to copy the data from the host
	cudaError(cudaMalloc((void**) &d_seq2, size_seq2), "Failed to copy seq2 from host to device - %s\n");
	cudaMemcpy(d_seq2, seq2, size_seq2, cudaMemcpyHostToDevice); //Copy seq2 from host to the GPU memory

	return d_seq2;
}


//Free allocated memory on GPU
void freeAllocated_matW(double *d_matW) 
{
	cudaError(cudaFree(d_matW), "Failed to free device matW - %s\n");
}

//Free allocated memory on GPU
void freeAllocated_seq1(char *d_seq1) 
{
	cudaError(cudaFree(d_seq1), "Failed to free device seq1 - %s\n");
}

//Free allocated memory on GPU
void freeAllocated_seq2(char *d_seq2) 
{
	cudaError(cudaFree(d_seq2), "Failed to free device seq2 - %s\n");
}


void cudaError(cudaError_t err, const char *messageError) 
{
	if (err != cudaSuccess) 
	{
		fprintf(stderr, messageError, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}
