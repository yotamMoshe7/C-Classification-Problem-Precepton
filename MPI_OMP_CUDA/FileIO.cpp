#include "FileIO.h"

PointsDataset loadPoints(FileInfo* info)
{
	int i, j;
	FILE* file = fopen(DATASET_PATH, "r");
	PointsDataset points;
	if (!file)
	{
		printf("File %s could not be opened", DATASET_PATH);
		exit(1);
	}
	fscanf(file, "%d %d %lf %lf %d %lf\n", &info->N, &info->K, &info->alpha0, &info->alphaMAX, &info->LIMIT, &info->QC);
	points.numOfPoints = info->N;
	points.dimensions = info->K;
	points.coords = (double*)calloc(info->N * info->K, sizeof(double));
	points.groupTags = (int*)calloc(info->N, sizeof(int));

	for (i = 0; i < info->N && feof(file) != EOF; ++i)
	{
		for (j = 0; j < info->K; ++j)
		{
			fscanf(file, "%lf ", &points.coords[(i * info->K) + j]);
		}
		fscanf(file, "%d", &points.groupTags[i]);
	}

	fclose(file);

	return points;
}


void writeResultsToFile(double alpha, double q, double* weights, 
	double w0, int k, double maxAlpha)
{
	int i;
	FILE* file = fopen(RESULTS_FILE_PATH, "w");

	if (!file || ferror(file))
	{
		printf("File %s could not be opened for writing", RESULTS_FILE_PATH);
		exit(1);
	}

	if (alpha >= maxAlpha)
	{
		fprintf(file, "Alpha was not found\n\n");
		printf("\nAlpha was not found\n\n", alpha, q);
	}

	fprintf(file, "Alpha minimum = %lf, q = %lf\n\n\n", alpha, q);
	printf("\nAlpha minimum = %lf, q = %lf\n\n\n", alpha, q);

	fprintf(file, "bias (w0) = %lf\n\n", w0);
	printf("bias (w0) = %lf\n\n", w0);

	for (i = 0; i < k; ++i)
	{
		fprintf(file, "W%d = %lf\n\n", i + 1, weights[i]);
		printf("W%d = %lf\n\n", i + 1, weights[i]);
	}

	fclose(file);
}