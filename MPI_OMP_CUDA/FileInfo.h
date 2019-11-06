#ifndef __FILE_INFO_H
#define __FILE_INFO_H

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "PointsDataset.h"

struct
{
	int N;
	int K;
	double alpha0;
	double alphaMAX;
	int LIMIT;
	double QC;
}
typedef FileInfo;

void createFileInfoType(MPI_Datatype* FileInfoType);

#endif // __FILE_INFO_H
