#ifndef __PERCEPTRON_CLASSIFICATION_H
#define __PERCEPTRON_CLASSIFICATION_H

#include <cmath>
#include <omp.h>
#include "FileInfo.h"
#include "FileIO.h"
#include "mpi.h"
#include "CudaMemPtr.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define MASTER 0

#define PAIR 2

extern cudaError_t trainWithCuda(CudaMemPtr* cudaData, const double* weights,
	double w0, double* results);

double prediction(const double* weights, double w0, const double * point, int dimensions);


void updateWeightsVector(double* weights, double* w0, double alpha, int sign,
	const double * point, int dimensions);


void trainPoints(const FileInfo* fileInfo, const PointsDataset* points,
	int numOfProcesses,
	int rank, CudaMemPtr* cudaData);

int classifyWithCuda(CudaMemPtr* cudaData, const PointsDataset* points,
	double* weights, double* w0, double alpha, double* calcResults);


int syncResults(double* myQAndAlpha, int numOfProcesses,
	const FileInfo* fileInfo, int rank, double* w0, double* weights,
	int* minAlphaRank, double* results);


int classifySequentialy(int n, const PointsDataset* points, double* weights,
	double* w0, double alpha, double* calcResults);

double qualityEvaluation(const PointsDataset* points, const double* calcResults);

#endif // !__PERCEPTRON_CLASSIFICATION_H
