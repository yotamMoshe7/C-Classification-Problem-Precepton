# Classification Problem Precepton
 implementation of binary classification problem using parallel computing with Cuda, OpenMP and MPI.
 The classification result must be smaller than the predefined QC variable.
 
 Parallel Solution
1.	The master job loads the tagged points dataset from file and other included info.
2.	The loaded points and data published to the other jobs using MPI, each process receives the whole dataset (Broadcast).
3.	Every process initiating a to a0* (rank + 1) and weights vector to 0’s.
4.	The classification calculation for every point done in parallel with CUDA, every point classification calculated using single kernel thread and activation function: f(x) = X * W
5.	Each classification result is checked until finding a misclassified point P.
6.	When a misclassified point P is found, the weights are updated using OMP threads, 
each thread calculates another weight: w[i] = w[i] + a * sign * P[i].
7.	Loop through steps 4-6 until none of the points were misclassified or LIMIT of iterations was reached.
8.	Using OMP, each process evaluates the quality he achieved: q = Nmis / N.
9.	All of the processes sharing their results (AllGather), checking if any of them has reached  the desired quality QC in parallel, or if any of them reached MAX.
10.	If any of the process reached QC, the other process terminated and the winning process sending his results to Master process using MPI_Send and MPI_Recv.
11.	Otherwise each process updates [a = a + a0 * numOfProcesses] and returns to step 4.
12.	At the end, Master process writes results to the results file.
