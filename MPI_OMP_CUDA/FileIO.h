
#ifndef __FILE_IO_H
#define __FILE_IO_H
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "PointsDataset.h"
#include "FileInfo.h"

#define DATASET_PATH "C:\\Users\\cudauser\\Downloads\\MPI_OMP_CUDA\\MPI_OMP_CUDA\\data1.txt"

#define RESULTS_FILE_PATH "C:\\Users\\cudauser\\Downloads\\MPI_OMP_CUDA\\MPI_OMP_CUDA\\results.txt"

PointsDataset loadPoints(FileInfo* info);


void writeResultsToFile(double alpha, double q, double* weights,
	double w0, int k, double maxAlpha);

#endif // !__FILE_IO_H
