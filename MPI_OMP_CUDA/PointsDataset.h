#ifndef __POINTS_DATASET_H

#define __POINTS_DATASET_H

struct
{
	int numOfPoints;
	int dimensions;
	double* coords;
	int* groupTags;
}
typedef PointsDataset;


const double * getPointAt(const PointsDataset* points, int position);
#endif // !__POINTS_DATASET_H
