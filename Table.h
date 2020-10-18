#ifndef TABLE_H
#define TABLE_H

#include <cuda_runtime.h>

#include <vector>

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


typedef std::vector<double> Row;

class Table {
public:
	Table(int n_rows);
	Table(int n_rows, int n_cols);
	~Table();

	double calculateValue(const double rowKey) const;
	double calculateValue(const double rowKey, const double columnKey) const;

	int getNumRows() const { return num_rows; }
	int getNumCols() const { return num_cols; }

	double operator()(unsigned int r, unsigned int c) const { return rows[r][c]; }

	Table& operator<<(const double n);

	double* allocate() const;

private:

	double linearInterpolation(const double x1, const double f_x1, const double x2,
		const double f_x2, const double x) const;

	std::vector<Row> rows;

	mutable double* d_vector;

	int rowCounter;
	int columnCounter;

	// Optimization
	mutable unsigned int last_row_index = 0;
	mutable unsigned int last_col_index = 0;

	size_t num_cols;
	size_t num_rows;
};

#endif
