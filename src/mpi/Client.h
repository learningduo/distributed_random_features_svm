#ifndef SVM_CLIENT_H
#define SVM_CLIENT_H


#include <iostream>
#include <armadillo>
#include <map>
#include "../algo/Svm.h"
#include "../parsers/BaseParser.h"
#include "../parsers/AdultParser.h"
#include <mpi.h>
#include "../utils/Normalizer.h"
#include "../utils/Configuration.h"

using namespace std;
using namespace arma;

extern mat rand_w;
extern mat bias;
extern bool t;

void mpiClients(BaseParser *dataParser, Configuration *config, const int dest = 0, const int tag = 0) {


    LOG(INFO) << "getting RF params from Server";

    // Fill RF params
    double *rand_w_arr = new double[config->rfDim * config->xDim];
    double *bias_arr = new double[config->rfDim];

    // receive random features parameters from server
    MPI_Recv(rand_w_arr, config->rfDim * config->xDim, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(bias_arr, config->rfDim, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // decode data
    rand_w = matFromArr(rand_w_arr, config->rfDim, config->xDim);
    bias = matFromArr(bias_arr, config->rfDim, 1);
    t = true;

    LOG(INFO) << "got RF params from Server";

    // Data initialization
    vec y;
    vec classes(config->classes);
//    mat x = normCol(dataParser->parse(y));
    mat x = dataParser->parse(y);

    // For running a single learning
    Svm svm;
    double* result = svm.runKernelRFSvmWrapper(x, x.n_rows, x.n_cols, y,
                                               classes, config->rfDim, config->learningRate, config->iterations,
                                               config->reg, config->rfParam);

    // send weight vector to server
    LOG(INFO) << "sending to server the learned weigths";
    MPI_Send(result, classes.size() * (config->rfDim + 1), MPI_DOUBLE, dest, tag, MPI_COMM_WORLD);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    LOG(INFO) << "rank: " << world_rank << " sent";

    // free allocated sapce
    delete(rand_w_arr);
    delete(bias_arr);
    delete(result);

}

#endif //SVM_CLIENT_H
