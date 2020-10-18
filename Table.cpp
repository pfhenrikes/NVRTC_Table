#include "Table.h"

#include <iostream>
#include <iterator>
#include <sstream>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cuda.h>

Table::Table(int n_rows) : num_rows(n_rows), num_cols(1) {
    rows.resize(n_rows);
    for (int i = 0; i < n_rows; i++) {
        rows[i].resize(2);
    }
    rowCounter = 0;
    columnCounter = 0;
    last_row_index = num_rows;
}

Table::Table(int n_rows, int n_cols)
        : num_rows(n_rows), num_cols(n_cols) {
    rows.resize(n_rows + 1);
    for (int i = 0; i <= n_rows; i++) {
        if (i == 0 && n_cols > 1) {
            rows[i].resize(n_cols);
        } else {
            rows[i].resize(n_cols + 1);
        }
    }
    rowCounter = 0;
    columnCounter = 0;
    last_row_index = num_rows;
    last_col_index = num_cols - 1;
}

Table::~Table() {
    auto code = cudaFree(d_vector);
    if (code != CUDA_SUCCESS) {
        std::cerr << "Error deleting table array!\n";
        exit(1);
    }
}

double Table::calculateValue(const double rowKey) const {
    unsigned int index = last_row_index;

    // if the key is off the end of the table, just return the
    // end-of-table value, do not extrapolate
    if (rowKey <= rows[0][0]) {
        return rows[0][1];
    } else if (rowKey >= rows[num_rows - 1][0]) {
        return rows[num_rows - 1][1];
    }

    while (index > 1 && rows[index - 1][0] > rowKey) {
        --index;
    }
    while (index < num_rows && rows[index][0] < rowKey) {
        ++index;
    }

    last_row_index = index;

    double factor = (rowKey - rows[index - 1][0]) / (rows[index][0] - rows[index - 1][0]);

    double value = rows[index - 1][1] + (rows[index][1] - rows[index - 1][1]) * factor;

    return value;
}

double Table::calculateValue(const double rowKey, const double columnKey) const {
    unsigned int rIndex = last_row_index;
    unsigned int cIndex = last_col_index;

    while (rIndex > 2 && rows[rIndex - 1][0] > rowKey) {
        --rIndex;
    }
    while (rIndex < num_rows && rows[rIndex][0] < rowKey) {
        ++rIndex;
    }

    while (cIndex > 1 && rows[0][cIndex - 1] > columnKey) {
        --cIndex;
    }
    while (cIndex < num_cols && rows[0][cIndex] < columnKey) {
        ++cIndex;
    }

    last_row_index = rIndex;
    last_col_index = cIndex;

    double q11 = rows[rIndex - 1][cIndex];
    double q12 = rows[rIndex - 1][cIndex + 1];
    double q21 = rows[rIndex][cIndex];
    double q22 = rows[rIndex][cIndex + 1];
    double x1 = rows[rIndex - 1][0];
    double x2 = rows[rIndex][0];
    double y1 = rows[0][cIndex - 1];
    double y2 = rows[0][cIndex];
    double x = rowKey;
    double y = columnKey;

    double R1 = linearInterpolation(x1, q11, x2, q21, x);
    double R2 = linearInterpolation(x1, q12, x2, q22, x);
    double value = linearInterpolation(y1, R1, y2, R2, y);

    return value;
}

double Table::linearInterpolation(const double x1, const double f_x1, const double x2,
                                  const double f_x2, const double x) const {
    return f_x1 + (f_x2 - f_x1) * (x - x1) / (x2 - x1);
}

Table &Table::operator<<(const double n) {
    rows[rowCounter][columnCounter] = n;
    if (num_cols > 1 && rowCounter == 0 && columnCounter + 1 == (int) num_cols) {
        columnCounter = 0;
        rowCounter++;
    } else if (columnCounter == (int) num_cols) {
        columnCounter = 0;
        rowCounter++;
    } else {
        columnCounter++;
    }

    return *this;
}

double *Table::allocate() const {
    std::vector<double> aux;

    for (auto row : rows) {
        aux.insert(aux.end(), row.begin(), row.end());
    }
//    for (auto el : aux) {
//        std::cout << el << " ";
//    }
//    std::cout << "\n";

    size_t size = aux.size();
//    std::cout << "Allocating " << size << " number of doubles!\n";
    auto code = cudaMalloc(&d_vector, size * sizeof(double));
    if (code != CUDA_SUCCESS) {
        std::cerr << "Error allocating device table!\n";
        exit(1);
    }
    cudaMemcpy(d_vector, aux.data(), size * sizeof(double), cudaMemcpyHostToDevice);
//    memcpy(d_vector, aux.data(), size * sizeof(double));

//    for(int i = 0; i<size; i++) {
//        std::cout << d_vector[i] << " ";
//    }
//    std::cout << "\n";

    return d_vector;
}
