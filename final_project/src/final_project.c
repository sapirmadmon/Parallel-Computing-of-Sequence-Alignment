//Sapir Madmon 209010230

#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>
#include "final_project.h"
#include <math.h>
#include <stdlib.h>
#include "cudaFunctions.h"


int main(int argc, char* argv[])
{

	int  my_id; 			/* rank of process */
	int  num_procs;       /* number of processes */


	/* start up MPI */
	MPI_Init(&argc, &argv);

	/* find out process rank */
	MPI_Comm_rank(MPI_COMM_WORLD, &my_id);

	/* find out number of processes */
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);


	//the program run with only 2 procs
	if (num_procs != 2)
	{
		printf("The program use only 2 processes!\n");
		MPI_Abort(MPI_COMM_WORLD, __LINE__);
	}


	Info info;
	double matWeight[NUM_ABC][NUM_ABC];
	const char* OUTPUT_FILE_NAME = "result.txt";
	const char* INPUT_FILE_NAME = "input.txt";

	MPI_Status status;

	FILE *fp = fopen(OUTPUT_FILE_NAME, "w"); //open output file to the result

	int index_offset, index_mutant;
	double maxScore = -INFINITY;
	double score;

	int lenSeq1, lenSeq2;
	int maxOffset;
	int numSeq2;

	double *d_matW;
	char *d_seq1, *d_seq2;

	double start_time = MPI_Wtime(); //start time

	if(my_id == ROOT)
		readFile(INPUT_FILE_NAME, &info); //only the master read from file all the data
	
	doBcast(&info, my_id); //broadcast to the slave the data
	fill_weight_mat(matWeight, info.weights); //Filling the weights matrix once at the beginning of the program


	d_seq1 = allocatedSeq1OnGPU(info.seq1); //allocate seq1 on GPU
	d_matW = allocatedWeightMat(matWeight);	//allocate weight matrix on GPU

	lenSeq1 = strlen(info.seq1);

	MPI_Datatype resultType;
	createTypeResult(&resultType);

	Result resultArrPerSeq[info.number_seq2]; //Array for storing the results of each seq2

	if(my_id == ROOT)
	{
		for (numSeq2 = 0; numSeq2 < info.number_seq2/2; numSeq2++) //The master work on half from all seq2's
		{
			maxScore = -INFINITY; //Initialize the initial value of maxScore

			lenSeq2 = strlen(info.seq2[numSeq2]);

			d_seq2 = allocatedSeq2OnGPU(info.seq2[numSeq2]); //allocate *specific* seq2 on GPU

			maxOffset = lenSeq1 - lenSeq2 - 1;

	#pragma omp parallel for
			for(index_offset = 0; index_offset <= maxOffset; index_offset ++)
			{
				score = findMaxScoreAndBestMutant(&info,lenSeq2,index_offset,numSeq2 ,&index_mutant, d_matW, d_seq1, d_seq2);
	#pragma omp critical
				if(score > maxScore)
				{
					maxScore = score;
					resultArrPerSeq[numSeq2].offset = index_offset;
					resultArrPerSeq[numSeq2].mutant = index_mutant;
					resultArrPerSeq[numSeq2].score = score;
				}
			}

			freeAllocated_seq2(d_seq2); //free allocated of seq2
		}
	}

	else //if (my_id == SLAVE)
	{
		int index;
		for (numSeq2 = info.number_seq2/2; numSeq2 < info.number_seq2; numSeq2++) //The slave work on the other half from all seq2's
		{
			maxScore = -INFINITY;

			lenSeq2 = strlen(info.seq2[numSeq2]);

			d_seq2 = allocatedSeq2OnGPU(info.seq2[numSeq2]); //allocate *specific* seq2 on GPU

			maxOffset = lenSeq1 - lenSeq2 - 1;

	#pragma omp parallel for
			for(index_offset = 0; index_offset <= maxOffset; index_offset ++)
			{
				score = findMaxScoreAndBestMutant(&info,lenSeq2,index_offset,numSeq2 ,&index_mutant, d_matW, d_seq1, d_seq2);
	#pragma omp critical
				if(score > maxScore)
				{
					maxScore = score;
					//The slave has its own array and the input of the results to the array will start
					//from the 0th place
					index = numSeq2 - (info.number_seq2/2);
					resultArrPerSeq[index].offset = index_offset;
					resultArrPerSeq[index].mutant = index_mutant;
					resultArrPerSeq[index].score = score;
				}
			}

			//Sending the result of a specific seq2 (by index) to the Master
			MPI_Send(&resultArrPerSeq[index], 1, resultType, ROOT, 0, MPI_COMM_WORLD);
			freeAllocated_seq2(d_seq2); //free allocated of seq2
		}
	}


	//Now, the master gets the results from the slave and consolidates all the results of all the seq2's into one arr.
	//he prints them to the console and to the output file
	if(my_id == ROOT)
	{
		int i;
		int numSeq2;

		for(i=info.number_seq2/2; i<info.number_seq2; i++)
		{
			//The Master gets the results of all the seq2s from the Slave
			//and puts them in their location in the original array (resultArrPerSeq)
			MPI_Recv(&resultArrPerSeq[i], 1, resultType, 1, 0, MPI_COMM_WORLD, &status);
		}


		//print the results to console
		for(numSeq2=0; numSeq2<info.number_seq2; numSeq2++)
			printf("Number of seq2: %d , The best Offset is: %d , The best mutant is: %d , The max Score: %f\n"
					,numSeq2, resultArrPerSeq[numSeq2].offset, resultArrPerSeq[numSeq2].mutant,
					resultArrPerSeq[numSeq2].score);

		//print the results to file
		for(numSeq2=0; numSeq2<info.number_seq2; numSeq2++)
			fprintf(fp, "Number of seq2: %d , The best Offset is: %d , The best mutant is: %d , The max Score: %f\n"
					,numSeq2, resultArrPerSeq[numSeq2].offset, resultArrPerSeq[numSeq2].mutant,
					resultArrPerSeq[numSeq2].score);
	}


	fclose(fp); //close result file

	freeAllAllocations(&info, d_seq1, d_matW); //free Allocations of matWeight, seq1 and seq2's 

	if(my_id == ROOT)
		printf("Total time: %lf\n",MPI_Wtime()-start_time);

	/* shut down MPI */
	MPI_Finalize();

	return 0;
}



int readFile (const char* fileName, Info* info)
{
	FILE *fp;
	int i;

	fp = fopen(fileName, "r");

	if(!fp)
	{
		printf("Error open file");
		return 0;
	}

	for(i = 0 ; i < NUM_WEIGHT ; i++)
	{
		if( i != NUM_WEIGHT - 1)
			fscanf(fp, "%f", &info->weights[i]);
		else
			fscanf(fp, "%f\n", &info->weights[i]);
	}

	fgets(info->seq1, SEQ1_MAX_SIZE, fp);

	fscanf(fp, "%d\n", &info->number_seq2);

	info->seq2 = (char**)malloc(sizeof(char*)*info->number_seq2);

	for(i = 0 ; i < info->number_seq2 ; i++)
	{
		info->seq2[i] = (char*)malloc(sizeof(char)*SEQ2_MAX_SIZE);
		fgets(info->seq2[i],SEQ2_MAX_SIZE, fp);
	}

	fclose(fp);
	return 1;
}


void createTypeResult(MPI_Datatype* resultType)
{
	int blocklen[RESULT_NUM_OF_ATTRIBUTES] = {1,1,1};
	MPI_Aint disp[RESULT_NUM_OF_ATTRIBUTES];
	MPI_Datatype types[RESULT_NUM_OF_ATTRIBUTES] = { MPI_INT, MPI_INT, MPI_DOUBLE};

	disp[0] = offsetof(Result, offset);
	disp[1] = offsetof(Result, mutant);
	disp[2] = offsetof(Result, score);

	MPI_Type_create_struct(RESULT_NUM_OF_ATTRIBUTES, blocklen, disp, types, resultType);
	MPI_Type_commit(resultType);
}


void fill_weight_mat(double matW[NUM_ABC][NUM_ABC], float weight[NUM_WEIGHT])
{
	int i, j;
	char ch1, ch2;

	for(i=0; i<NUM_ABC; i++)
	{
		ch1 = i + 'A';

		for(j=0; j<NUM_ABC; j++)
		{
			ch2 = j + 'A';

			if(ch1 == ch2)
				matW[i][j] = weight[0];

			else if(checkExistingInConservative(ch1, ch2) == 1)
				matW[i][j] = -weight[1];

			else if(checkExistingInSemiConservative(ch1, ch2) == 1)
				matW[i][j] = -weight[2];

			else matW[i][j] = -weight[3];
		}
	}
}



void freeAllAllocations(Info *info, char *d_seq1, double *d_matW) //free GPU and CPU allocations
{
	int i;
	for (i = 0; i < info->number_seq2; ++i)
		free(info->seq2[i]);

	free(info->seq2);
	freeAllocated_seq1(d_seq1);
	freeAllocated_matW(d_matW);
}



double findMaxScoreAndBestMutant(Info* info,int lenSeq2,int offset,int numSeq2,int * bestMutant, double *d_matW, char *d_seq1, char *d_seq2)
{
	int i;
	double maxScore = -INFINITY;
	double *arrSorcePerSeq;

	//get all mutants results from GPU to seq2 for a specific offset
	arrSorcePerSeq = computeArrScoreOnGPU(d_matW, d_seq1, d_seq2, offset, numSeq2, lenSeq2);

	for (i = 0; i < lenSeq2; i++)  //find the best mutant and max score at arrScorePerSeq
	{
		if (arrSorcePerSeq[i] > maxScore)
		{
			maxScore = arrSorcePerSeq[i];
			*bestMutant = i + 1; 
		}
	}

	free(arrSorcePerSeq);
	return maxScore;
}


void doBcast(Info * info, int my_id)
{
	int i;

	MPI_Bcast(info->weights,NUM_WEIGHT,MPI_FLOAT,ROOT,MPI_COMM_WORLD);	//Bcast to the weights
	MPI_Bcast(info->seq1,SEQ1_MAX_SIZE,MPI_CHAR,ROOT,MPI_COMM_WORLD);	//Bcast to the seq1
	MPI_Bcast(&info->number_seq2,1,MPI_INT,ROOT,MPI_COMM_WORLD); 		//Bcast to the num of seq2

	if(my_id != ROOT)
	{
		info->seq2 = (char**)malloc(sizeof(char*)*info->number_seq2);

		for(i = 0 ; i < info->number_seq2 ; i++)
			info->seq2[i] = (char*)malloc(sizeof(char)*SEQ2_MAX_SIZE);
	}

	for(i = 0 ; i < info->number_seq2 ; i++)
		MPI_Bcast(info->seq2[i],SEQ2_MAX_SIZE,MPI_CHAR,ROOT,MPI_COMM_WORLD); //Bcast to all seq2s
}


int checkExistingInConservative (char ch1, char ch2) //check if existing in Conservative Group
{
	int i;

	for(i=0; i<9; i++)
	{
		if(checkTwoCharsIntoGroup(ConservativeGroups[i], ch1, ch2) == 1)
			return 1;
	}
	return 0;
}


int checkExistingInSemiConservative (char ch1, char ch2) //check if existing in Seami-Conservative Group
{
	int i;

	for(i=0; i<11; i++)
	{
		if(checkTwoCharsIntoGroup(SemiConservativeGroups[i], ch1, ch2) == 1)
			return 1;
	}
	return 0;
}


int checkTwoCharsIntoGroup(const char* str, char ch1, char ch2) //check if two chars in the same group
{
	int char1 = checkCharInsideString(str, ch1);
	int char2 = checkCharInsideString(str, ch2);

	if((char1 == 1) && (char2 == 1))
		return 1;
	else
		return 0;

}

int checkCharInsideString (const char* str, char ch) //check if char inside string
{
	if(strchr(str, ch))
		return 1;
	else
		return 0;
}
