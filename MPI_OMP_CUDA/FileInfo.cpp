#include "FileInfo.h"

void createFileInfoType(MPI_Datatype* FileInfoType)
{
	MPI_Aint disp[6];
	int blocklen[] = { 1, 1, 1, 1, 1, 1, 1 };
    MPI_Datatype type[] = {MPI_INT, MPI_INT, MPI_DOUBLE, 
		MPI_DOUBLE, MPI_INT, MPI_DOUBLE};

    disp[0] = offsetof(FileInfo, N);
    disp[1] = offsetof(FileInfo,K);
    disp[2] = offsetof(FileInfo,alpha0);
    disp[3] = offsetof(FileInfo, alphaMAX);
    disp[4] = offsetof(FileInfo, LIMIT);
    disp[5] = offsetof(FileInfo, QC);
    MPI_Type_create_struct(6, blocklen, disp, type, FileInfoType);
    MPI_Type_commit(FileInfoType);
}
