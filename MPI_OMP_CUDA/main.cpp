#define _CRT_SECURE_NO_WARNINGS
#include "mpi.h"
#include "CudaMemPtr.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "FileInfo.h"
#include "FileIO.h"
#include "PerceptronClassification.h"

#define MASTER 0

extern void cudaAllocationAndCpy(CudaMemPtr* cudaData, const double* coords, 
	int numOfPoints, int dimensions);

extern void freeCudaMem(CudaMemPtr* cudaData);

int main(int argc, char* argv[])
{
    int rank, numprocs;
    double startTime, endTime;
    FileInfo fileInfo;
	PointsDataset points;
	MPI_Datatype FileInfoType;
	CudaMemPtr cudaData;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    createFileInfoType(&FileInfoType);

    if(rank == MASTER)
    {
        points = loadPoints(&fileInfo);
    }

    MPI_Bcast(&fileInfo, 1, FileInfoType, MASTER, MPI_COMM_WORLD);

    if(rank != MASTER)
    {
        points.numOfPoints = fileInfo.N;
        points.dimensions = fileInfo.K;
        points.coords = (double*)calloc(fileInfo.N * fileInfo.K, sizeof(double));
        points.groupTags = (int*)calloc(fileInfo.N, sizeof(int));
    }

    MPI_Bcast(points.coords, fileInfo.N * fileInfo.K, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(points.groupTags, fileInfo.N, MPI_INT, MASTER, MPI_COMM_WORLD);

	cudaAllocationAndCpy(&cudaData, 
		points.coords, 
		points.numOfPoints, 
		points.dimensions);

	startTime = MPI_Wtime();
    trainPoints(&fileInfo, &points, numprocs, rank, &cudaData);
    endTime = MPI_Wtime();
	freeCudaMem(&cudaData);
	
    if(rank == MASTER)
    {
        printf("\ntotal time: %lf\n", endTime - startTime);
		fflush(stdout);
    }
    MPI_Finalize();

    free(points.coords);
    free(points.groupTags);
    return 0;
}
