#ifndef SVM_UTILS_H
#define SVM_UTILS_H

#include <armadillo>
#include <map>

#include "../../extern/easylogging++.h"

using namespace arma;
using namespace std;


// Returns a map of weights
// Getting a pointer to double, which is what the server receives,
// the number of classes (class_size), and the dimension (col)
// Every (col + 1) indexes, the first item is the class id, and the rest are their weights.
map<double, vec> static arrToMap(double *result, const int class_size, const int col) {
    map<double, vec> ans;

    for(int i = 0; i < class_size ; ++i) {
        double tempClass = result[i * (col + 1)];
        vec tempVec(col);
        for(int j = 1; j <=col; ++j) {
            tempVec(j-1) = result[i*(col+1) + j];
        }
        ans[tempClass] = tempVec;
    }

    return ans;
}


// Returns a pointer to double array.
// Getting a matrix
// Converting the matrix to a double array so we could transfer it over mpi.
double static *arrFromMat(mat x) {
    double *arr = new double[x.n_rows * x.n_cols];
    for (int i = 0; i < x.n_rows; i++) {
        for (int j = 0; j < x.n_cols; j++) {
            arr[i*x.n_cols + j] = x(i, j);
        }
    }
    return arr;
}

// Returns a matrix.
// Getting a pointer to double array, the number of rows and cols.
// Converting a double array so we could get back the matrix.
mat static matFromArr(double *x, const int row, const int col) {
    mat X = mat(row, col);
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            X(i, j) = x[i*col + j];
        }
    }
    return X;
}

#endif //SVM_UTILS_H
