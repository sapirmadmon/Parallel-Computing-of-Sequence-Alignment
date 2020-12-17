# Parallel Implementation Of Sequence alignment
###

## Problem definition:
Let Seq1 and Seq2 be a given Sequences of letters.
For all Mutant Sequences of Seq2 find an offset n which produce a maximum Alignment Score
against Seq1. Among all results, choose the Mutant Sequence MS(k) with the best Alignment Score

## Solution:
Parallelizing the algorithm:
## MPI -
The MPI is performed using in a static method (I preferred to use this method because we know the number of seq2 from the input file) <br>

o	The program ran with 2 processes, Master (root) and Slave. <br>
o	In the first step, the master reads the input file and sends the data to the slave via broadcast. <br>

o	Next, half of the strings (the seq2's) the master takes to calculate, and the remaining half the slave takes. <br>

o	When the slave finishes calculating the results, he sends them to the master. <br>
o	The master received the results from the slave (and in addition, he also has the results he calculated himself) <br>

o	now, the master can print the results (to the output file and the console) <br>
I chose to parallelize the strings of seq2 with MPI because each seq has independent calculations. In addition, each process has a lot of calculations to do and it's fit to use process or a different computer for this heavy task and not threads. 

##	OMP - parallel offsets calculation
o	Using the Omp library I parallelized the task of calculating all the offsets. <br>
It's a big array size that should be parallel. It's a loop that calculates the bast offset and the best result. Therefore, the omp fits perfectly for this task.

##	CUDA -
o	Using Cuda I parallelized the task of calculating the score of each mutant. There are a lot of mutants, and each mutant requires a simple calculation – and Cuda fit for this task. As we learned in class, with Cuda we can use a lot of threads and give each thread a simple tiny task. <br>

o	What GPU returns to CPU is an array of all scores that calculated in parallel. <br>

o	In addition, at class we also learned that copying the data from a host (CPU) to the device (GPU) is a time-consuming task, therefore, in order to use Cuda efficiently, I initialized Cuda data as the first task of the slave and master with a constant data (the matrix of weighs and the seq1) – that’s lead to copy the data from the host to the device only once. <br>

