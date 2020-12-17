//Sapir Madmon 209010230

#include <stddef.h>

//includes for vLab afeka
#include "/usr/local/cuda-9.1/include/cuda.h"
#include "/usr/local/cuda-9.1/include/cuda_runtime_api.h"


#ifndef CUDAUTILITY_H_
#define CUDAUTILITY_H_


#define NUM_ABC 26

double* computeArrScoreOnGPU(double *d_matW, char *seq1, char *seq2, int offSet, int numSeq2, int seq2Len);

double* allocatedWeightMat(double matW[NUM_ABC][NUM_ABC]);
char* allocatedSeq1OnGPU(char *seq1);
char* allocatedSeq2OnGPU(char *seq2);

void freeAllocated_matW(double *d_matW);
void freeAllocated_seq1(char *d_seq1);
void freeAllocated_seq2(char *d_seq2);

void cudaError(cudaError_t err, const char *messageError);

#endif /* CUDAUTILITY_H_ */
