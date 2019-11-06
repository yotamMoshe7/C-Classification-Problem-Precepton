#include "PointsDataset.h"

//each point takes 'dimension' doubles
const double * getPointAt(const PointsDataset* points, int position)
{
    return points->coords + (points->dimensions * position);
}