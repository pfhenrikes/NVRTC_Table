#pragma once
#ifndef DEVICE_TABLE_H
#define DEVICE_TABLE_H

#include <cuda_runtime.h>

#define CUDA_SAFE_CALL(x)                                         \
  do {                                                            \
    CUresult result = x;                                          \
    if (result != CUDA_SUCCESS) {                                 \
      const char *msg;                                            \
      cuGetErrorName(result, &msg);                               \
      std::cerr << "\nerror: " #x " failed with error "           \
                << msg << '\n';                                   \
      exit(1);                                                    \
    }                                                             \
  } while(0)


class DeviceTable {
public:
    __host__ __device__ DeviceTable(int n_rows, double *rows) : num_rows(n_rows), num_cols(1), last_row_index(n_rows),
                                                                last_col_index(0), rows(rows) {}

    __host__ __device__ DeviceTable(int n_rows, int n_cols, double *rows) : num_rows(n_rows), num_cols(n_cols + 1),
                                                                            last_row_index(n_rows),
                                                                            last_col_index(n_cols - 1),
                                                                            rows(rows) {}

    __host__ __device__ ~DeviceTable() {}

    __host__ __device__ double calculateValue(const double rowKey) const {
        unsigned int index = last_row_index;

        // if the key is off the end of the table, just return the
        // end-of-table value, do not extrapolate
        if (rowKey <= rows[0]) {
            return rows[1];
        } else if (rowKey >= rows[(num_rows - 1) * 2]) {
            return rows[(num_rows - 1) * 2 + 1];
        }

        while (index > 1 && rows[(index - 1) * 2] > rowKey) {
            --index;
        }
        while (index < num_rows && rows[index * 2] < rowKey) {
            ++index;
        }

        last_row_index = index;

        double factor = (rowKey - rows[(index-1) * 2]) / (rows[index * 2] - rows[(index-1) * 2]);

        double value = rows[(index-1) * 2 + 1] + (rows[index * 2 + 1] - rows[(index-1) * 2 + 1]) * factor;

        return value;
    }

    __host__ __device__ double calculateValue(const double rowKey, const double columnKey) const {
        unsigned int rIndex = last_row_index;
        unsigned int cIndex = last_col_index;

        while (rIndex > 2 && rows[(rIndex - 1) * num_cols - 1] > rowKey) {
            --rIndex;
        }
        while (rIndex < num_rows && rows[(rIndex) * num_cols - 1] < rowKey) {
            ++rIndex;
        }

        while (cIndex > 1 && rows[cIndex - 1] > columnKey) {
            --cIndex;
        }
        while (cIndex < num_cols && rows[cIndex] < columnKey) {
            ++cIndex;
        }

        last_row_index = rIndex;
        last_col_index = cIndex;

        double q11 = rows[(rIndex - 1) * num_cols + cIndex - 1];
        double q12 = rows[(rIndex - 1) * num_cols + cIndex];
        double q21 = rows[rIndex * num_cols + cIndex - 1];
        double q22 = rows[rIndex * num_cols + cIndex];
        double x1 = rows[(rIndex - 1) * num_cols - 1];
        double x2 = rows[rIndex * num_cols - 1];
        double y1 = rows[cIndex - 1];
        double y2 = rows[cIndex];
        double x = rowKey;
        double y = columnKey;

        double R1 = linearInterpolation(x1, q11, x2, q21, x);
        double R2 = linearInterpolation(x1, q12, x2, q22, x);
        double value = linearInterpolation(y1, R1, y2, R2, y);

        return value;
    }

    __host__ __device__ int getNumRows() const { return num_rows; }

    __host__ __device__ int getNumCols() const { return num_cols - 1; }

    __host__ __device__ double operator()(unsigned int r, unsigned int c) const { return rows[r * num_cols + c]; }

private:
    __host__ __device__ double linearInterpolation(const double x1, const double f_x1, const double x2,
                                                   const double f_x2, const double x) const {
        return f_x1 + (f_x2 - f_x1) * (x - x1) / (x2 - x1);
    }

    double *rows;

    // Optimization
    mutable unsigned int last_row_index;
    mutable unsigned int last_col_index;

    size_t num_cols;
    size_t num_rows;
};

#endif
