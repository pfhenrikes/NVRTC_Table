#include "DeviceTable.h"
__global__ void getValueOneCol(DeviceTable* table, double* result) {
	*result = table->calculateValue(1.9);
}
__global__ void getValueMultipleCols(DeviceTable* table, double* result) {
    *result = table->calculateValue(1.5, 3.5);
}