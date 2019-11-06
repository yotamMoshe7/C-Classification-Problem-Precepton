
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "CudaMemPtr.h"

#include <stdio.h>

cudaError_t trainWithCuda(CudaMemPtr* cudaMem, const double* weights,
	double w0, double* results);

void cudaAllocationAndCpy(CudaMemPtr* cudaMem, const double* coords, int numOfPoints, int dimensions);

void checkErrors(cudaError_t cudaStatus, CudaMemPtr* cudaMem, const char* errorMessage);

void freeCudaMem(CudaMemPtr* cudaMem);


__global__ void trainKernel(double *coords, const double * weights, double w0,
	double *results, int dimensions, int numOfPoints, int pointsPerBlock);

__global__ void trainKernel(double *coords, const double * weights, double w0,
	double *results, int dimensions, int numOfPoints, int pointsPerBlock)
{
	int id = blockIdx.x * pointsPerBlock + threadIdx.x, i;

	if (blockIdx.x == gridDim.x - 1 && numOfPoints % blockDim.x <= threadIdx.x)
		return;

	results[id] = w0;

	for (i = 0; i < dimensions; i++)
	{
		results[id] += coords[id*dimensions + i] * weights[i];
	}
}

void cudaAllocationAndCpy(CudaMemPtr* cudaMem, const double* coords, int numOfPoints, int dimensions)
{
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	checkErrors(cudaStatus, cudaMem,"cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");

	cudaStatus = cudaMalloc((void**)&cudaMem->cudaCoords, numOfPoints * dimensions * sizeof(double));
	checkErrors(cudaStatus, cudaMem, "cudaMalloc failed! cudaCoords");

	cudaStatus = cudaMalloc((void**)&cudaMem->cudaWeights, dimensions * sizeof(double));
	checkErrors(cudaStatus, cudaMem,"cudaMalloc failed! cudaWeights");

	cudaStatus = cudaMalloc((void**)&cudaMem->cudaResults, numOfPoints * sizeof(double));
	checkErrors(cudaStatus, cudaMem, "cudaMalloc failed! cudaWeights");

	cudaStatus = cudaMemcpy(cudaMem->cudaCoords, coords, numOfPoints * dimensions * sizeof(double), cudaMemcpyHostToDevice);
	checkErrors(cudaStatus, cudaMem, "cudaMemcpy failed! cudaCoords");

	cudaMem->numOfPoints = numOfPoints;
	cudaMem->dimensions = dimensions;
}

cudaError_t trainWithCuda(CudaMemPtr* cudaMem, const double* weights, 
	double w0, double* results)
{
	double* cudaCoords = cudaMem->cudaCoords;
	double* cudaWeights = cudaMem->cudaWeights;
	double* cudaResults = cudaMem->cudaResults;
	int numOfPoints = cudaMem->numOfPoints;
	int dimensions = cudaMem->dimensions;
	cudaError_t cudaStatus;
	int numOfBlocks, remainBlock, pointsPerBlock;
	cudaDeviceProp prop;
	cudaMem->w0 = w0;

	cudaGetDeviceProperties(&prop, 0);

	cudaStatus = cudaMemcpy(cudaMem->cudaWeights, weights, dimensions * sizeof(double), cudaMemcpyHostToDevice);
	checkErrors(cudaStatus, cudaMem,
		"cudaMemcpy failed! cudaWeights");

	pointsPerBlock = prop.maxThreadsPerBlock;
	numOfBlocks = numOfPoints / prop.maxThreadsPerBlock;
	remainBlock = numOfPoints % prop.maxThreadsPerBlock != 0;
	trainKernel<<<numOfBlocks + remainBlock, pointsPerBlock >>>(cudaMem->cudaCoords, cudaWeights, w0, cudaResults, dimensions, numOfPoints, pointsPerBlock);

	cudaStatus = cudaGetLastError();
	checkErrors(cudaStatus, cudaMem,
		"classifyKernel failed!");

	cudaStatus = cudaDeviceSynchronize();
	checkErrors(cudaStatus, cudaMem,
		"cudaDeviceSynchronize failed after classifyKernel!");

	cudaStatus = cudaMemcpy(results, cudaResults, numOfPoints * sizeof(double), cudaMemcpyDeviceToHost);
	checkErrors(cudaStatus, cudaMem,
		"cudaMemcpy failed! results");

	return cudaStatus;
}

void freeCudaMem(CudaMemPtr* cudaMem)
{
	cudaFree(cudaMem->cudaCoords);
	cudaFree(cudaMem->cudaWeights);
	cudaFree(cudaMem->cudaResults);
}

void checkErrors(cudaError_t cudaStatus, 
	CudaMemPtr* cudaMem, const char* message)
{
	if (cudaStatus != cudaSuccess) 
	{
		fprintf(stderr, message);
		cudaFree(cudaMem->cudaCoords);
		cudaFree(cudaMem->cudaWeights);
		cudaFree(cudaMem->cudaResults);
	}
}
