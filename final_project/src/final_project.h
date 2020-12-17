//Sapir Madmon 209010230

#ifndef FINAL_PROJECT_H_
#define FINAL_PROJECT_H_

#define NUM_WEIGHT 4
#define SEQ1_MAX_SIZE 3000
#define SEQ2_MAX_SIZE 2000
#define NUM_ABC 26
#define ROOT 0
#define SLAVE 1
#define RESULT_NUM_OF_ATTRIBUTES 3



typedef struct {
	float weights[NUM_WEIGHT];
	char seq1[SEQ1_MAX_SIZE];
	int number_seq2;
	char** seq2;
}Info;


typedef struct {
	int offset;
	int mutant;
	double score;
}Result;


const char* ConservativeGroups[9] =
			{"NDEQ", "NEQK", "STA",
			"MILV", "QHRK", "NHQK",
			"FYW", "HY", "MILF"};

const char* SemiConservativeGroups[11] =
			{"SAG", "ATV", "CSA",
			"SGND", "STPA", "STNK",
			"NEQHRK", "NDEQHK", "SNDEQK",
			"HFY", "FVLIM"};


int readFile (const char* fileName, Info* info);

int checkExistingInConservative (char ch1, char ch2);
int checkExistingInSemiConservative (char ch1, char ch2);
int checkTwoCharsIntoGroup(const char* str, char ch1, char ch2);
int checkCharInsideString (const char* str, char ch);

void fill_weight_mat(double matW[NUM_ABC][NUM_ABC], float weight[NUM_WEIGHT]);

double findMaxScoreAndBestMutant(Info* info,int lenSeq2,int offset,int numSeq2,int * bestMutant, double *d_matW, char *d_seq1, char *d_seq2);

void doBcast(Info * info,int my_id);

void createTypeResult(MPI_Datatype* resultType);

void freeAllAllocations(Info *info, char *d_seq1, double *d_matW);


#endif /* FINAL_PROJECT_H_ */
