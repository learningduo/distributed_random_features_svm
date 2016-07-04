#ifndef SVM_H
#define SVM_H

#include <iostream>
#include <vector>
#include <armadillo>
#include <map>
#include <random>
#include "../utils/Utils.h"

#include "../../extern/easylogging++.h"

//#define ARMA_DONT_USE_WRAPPER

using namespace std;
using namespace arma;


class Svm {

public:


    map<double, vec> runLinearSvm(vector<vec> x, vector<double> y, vector<double> classes, const double learningRate = 0.01,
                                  const int numIterations = 100, const double regParam = 0.00001);

    map<double, vec> runKernelRFSvm(mat X, vec y, vec classes, const int rfDim, const double learningRate = 0.01,
                                    const int numIterations = 20, const double regParam = 0.00001, double sigma = 1.0);

    double *runKernelRFSvmWrapper(mat X, int row, int col, vec y, vec classes,
                                  const int rfDim, const double learningRate = 0.01,
                                  const int numIterations = 20, const double regParam = 0.00001,
                                  double sigma = 19.0);


    double static argmax(map<double, vec> wList, vec x_i, double y_i);

    mat static randomFeatures(mat X, int rfDim, double sigma);

private:
    void learningUpdate(map<double, vec> &wList, vec classes,
                                    const double learningRate, const double regParam, vec x_i, double y_i, double y_hat);

    void epochLearn(mat x, vec y, map<double, vec> &wList, vec classes,
                                const double learningRate, const double regParam);
};


#endif //SVM_H
