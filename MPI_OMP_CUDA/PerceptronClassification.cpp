#include "PerceptronClassification.h"

int classifyWithCuda(CudaMemPtr* cudaData,
	const PointsDataset* points,
	double* weights,
	double* w0,
	double alpha, double* calcResults)
{
	int i;
//	double* calcResults = (double*)calloc(points->numOfPoints, sizeof(double));
	trainWithCuda(cudaData, weights, *w0, calcResults);

	for (i = 0; i < points->numOfPoints; i++)
	{
		int sign = calcResults[i] > 0 ? 1: -1;

		const double* point = getPointAt(points, i);
		if (points->groupTags[i] != sign)
		{
			updateWeightsVector(weights, w0, alpha, points->groupTags[i], point, points->dimensions);
			return 1;
		}
	}
	return 0;
}

int syncResults(double* myQAndAlpha, int numprocs, const FileInfo* fileInfo,
	int rank, double* w0, double* weights, int* minAlphaRank,
	double* results)
{
	const int NO_ONE = -1;
	int i, dimensions = fileInfo->K, finished = 0;;
	double* allResultsBuffer = (double*)calloc(numprocs * PAIR, sizeof(double));

	MPI_Allgather(myQAndAlpha, PAIR, MPI_DOUBLE, allResultsBuffer, PAIR,
		MPI_DOUBLE, MPI_COMM_WORLD);

	*minAlphaRank = NO_ONE;

	for (i = 0; i < numprocs; ++i)
	{
		if (allResultsBuffer[i * PAIR] < fileInfo->QC ||
			allResultsBuffer[i*PAIR + 1] > fileInfo->alphaMAX - fileInfo->alpha0)
		{
			*minAlphaRank = i;
			break;
		}
	}

	if (MASTER != *minAlphaRank && *minAlphaRank != NO_ONE) {
		if (rank == *minAlphaRank)
		{
			MPI_Send(w0, 1, MPI_DOUBLE, MASTER, 0, MPI_COMM_WORLD);
			MPI_Send(weights, dimensions, MPI_DOUBLE, MASTER, 0, MPI_COMM_WORLD);
		}
		else if (rank == MASTER)
		{
			MPI_Recv(w0, 1, MPI_DOUBLE, *minAlphaRank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Recv(weights, dimensions, MPI_DOUBLE, *minAlphaRank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
	}
	results[0] = allResultsBuffer[PAIR* (*minAlphaRank)];
	results[1] = allResultsBuffer[PAIR* (*minAlphaRank) + 1];
	free(allResultsBuffer);
	return *minAlphaRank != NO_ONE;
}

void trainPoints(const FileInfo* fileInfo, const PointsDataset* points,
	int numprocs, int rank, CudaMemPtr* cudaData)
{
	const int NO_ONE = -1;
	int i, wrong, dimensions = fileInfo->K, minAlphaRank = NO_ONE;
	double alpha = fileInfo->alpha0 * (1 + rank), quality;
	double w0 = 0, *weights = (double*)calloc(dimensions, sizeof(double));
	double* calcResults = (double*)calloc(fileInfo->N, sizeof(double));
	double qualityAlphaPair[PAIR], results[PAIR];

	if (rank == MASTER)
	{
		printf("\nDesired Quality = %lf aMax = %lf\n\n", fileInfo->QC, fileInfo->alphaMAX);
		fflush(stdout);
	}

	do {
		int iteration = 0;
		do {
			wrong = classifyWithCuda(cudaData, points, weights, &w0, alpha, calcResults);
			//classifySequentialy(points->numOfPoints, points, weights, &w0, alpha, calcResults);
			iteration++;
		} while (wrong && iteration < fileInfo->LIMIT);

		quality = qualityEvaluation(points, calcResults);
		qualityAlphaPair[0] = quality;
		qualityAlphaPair[1] = alpha;
		alpha += fileInfo->alpha0 * numprocs;

		syncResults(qualityAlphaPair, numprocs, fileInfo, rank,
			&w0, weights, &minAlphaRank, results);

	} while (quality > fileInfo->QC &&
		qualityAlphaPair[1] < fileInfo->alphaMAX
		&& minAlphaRank == NO_ONE);

	alpha -= fileInfo->alpha0 * numprocs;

	if (rank == MASTER)
	{
		double minAlpha = minAlphaRank == NO_ONE ? alpha : results[1];
		quality = minAlphaRank == NO_ONE ?
			quality : results[0];
		
		writeResultsToFile(minAlpha, quality, weights, w0, 
			dimensions, fileInfo->alphaMAX);
	}

	free(weights);
	free(calcResults);
}

void updateWeightsVector(double* weights, double* w0, double alpha, int sign, const double * point, int k)
{
	int i;

#pragma omp parallel for
	for (i = 0; i < k; ++i)
	{
		weights[i] += (alpha * sign) * point[i];
	}
	*w0 += (alpha * sign);
}


// Used only when cancelling CUDA
double prediction(const double* weights, double w0, const double * point, int k)
{
	int i;
	double result = 0;
	for (i = 0; i < k; ++i)
	{
		result += weights[i] * point[i];
	}
	result += w0;
	return result;
}

// Used when cuda is cancelled
int classifySequentialy(int n, const PointsDataset* points, double* weights,
	double* w0, double alpha, double* calcResults)
{
	int i, k = points->dimensions;

#pragma omp parallel for
	for (i = 0; i < n; ++i)
	{
		const double* point = getPointAt(points, i);
		calcResults[i] = prediction(weights, *w0, point, k);
	}

	for (i = 0; i < n; i++)
	{
		int sign = calcResults[i] > 0 ? 1 : -1;
		const double* point = getPointAt(points, i);

		if (sign != points->groupTags[i])
		{
			updateWeightsVector(weights, w0, alpha, points->groupTags[i], 
				point, k);
			return 1;
		}
	}
	return 0;
}

double qualityEvaluation(const PointsDataset* points, const double* calcResults)
{
	double misclassified = 0;
	int i;

#pragma omp parallel for reduction(+:misclassified)
	for (i = 0; i < points->numOfPoints; i++)
	{
		int sign = calcResults[i] > 0 ? 1 : -1;
		if (points->groupTags[i] != sign)
		{
			misclassified++;
		}
	}
	return misclassified / points->numOfPoints;

}