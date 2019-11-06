#ifndef __CUDA_MEM_PTR_H
#define __CUDA_MEM_PTR_H

struct
{
	int numOfPoints;
	int dimensions;
	double w0;
	double* cudaCoords;
	double* cudaWeights;
	double* cudaResults;
}
typedef CudaMemPtr;

#endif // !__CUDA_MEM_PTR_H

