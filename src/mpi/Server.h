#ifndef SVM_SERVER_H
#define SVM_SERVER_H


#include <iostream>
#include <armadillo>
#include <map>
#include <mpi.h>
#include "../algo/Svm.h"
#include "../parsers/BaseParser.h"
#include "../utils/Normalizer.h"
#include "../utils/Configuration.h"


using namespace std;
using namespace arma;

extern mat rand_w;
extern mat bias;
extern bool t;


// Returns the accuracy of the model
// Getting the weigths of the model and the x & y of the test set
// Comparing the prediction of the model comparing it's true values.
double validate(map<double, vec> svm_train, mat xTest, vec yTest) {

    int correct = 0, overall = 0;
    LOG(DEBUG) << "Validation";
    LOG(DEBUG) << "vec length: " << svm_train.at(1).size() << ", mat length: " << xTest.n_rows << " " << xTest.n_cols;

    for (int i = 0; i < xTest.n_rows; i++) {
        vec curX = xTest.row(i).t();
        double y_hat = Svm::argmax(svm_train, curX, yTest(i));
        if(y_hat == yTest(i)) {
            correct++;
        }
        overall++;
    }

    double acc = ((double) correct)/overall;
    LOG(INFO) << "Model accuracy: " << acc;
    return acc;
}



//
// Getting the a vector of map which is the weigths of the different servers, and a
// map which the answer will get inside
// Making a combinationg of the different results from the different clients.
// Based on average
void weigthsCombine(vector<map<double, vec>> res, map<double, vec> &ans) {
    ans = res.at(0);

    for (int i = 1; i < res.size(); i++) {
        for (map<double, vec>::iterator iter = res.at(i).begin(); iter != res.at(i).end(); iter++) {
            ans.at(iter->first) += iter->second;
        }
    }

    // division
    for (map<double, vec>::iterator iter = ans.begin(); iter != ans.end(); iter++) {
        ans.at(iter->first) /= res.size();
    }
};


void fillRFParams(Configuration *config) {
    int D = config->rfDim;
    int NProj = config->xDim;

    // run random initialization only once, 't' is a global variable
    if (!t) {
        rand_w = mat(D, NProj, fill::randn); // create random w
        rand_w *= sqrt(2 * config->rfParam);
        bias = mat(D, 1, fill::randu);
        bias *= 2.0 * M_PI;
        t = true;
    }
}


void mpiServer(BaseParser *dataParser, Configuration *config) {

    // Just for the normalization vector. // TODO - find another and better option
    vec y;
    //normMat(Parser::parseAdult(y, "resources/adult/a_train.csv", "resources/adult/a_ytrain.csv"));

    // Parameters handling...
    fillRFParams(config);
    LOG(INFO) << "sending RF params to clients";
    // Sending
    double *rand_w_send = arrFromMat(rand_w);
    double *bias_send = arrFromMat(bias);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // sending random features parameters to clients
    for (int i = 1; i < world_size; i++) {
        MPI_Send(rand_w_send, config->rfDim * config->xDim, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
        MPI_Send(bias_send, config->rfDim, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
    }
    LOG(INFO) << "RF params sent";

    // allocate space for receiving weight vectors from clients
    int size = (config->classes).size() * (config->rfDim + 1);
    int n_clients = world_size - 1;
    double **X = new double*[n_clients];
    for (int i = 0; i < n_clients; i++) {
        X[i] = new double[size];
    }
    LOG(INFO) << "amount of receiving servers: " << n_clients << ", size of arrays: " << size;

    // receive weight vector from client
    for (int i = 1; i <= n_clients; i++) {
        MPI_Recv(X[i - 1], size, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        LOG(INFO) << "got " << i;
    }
    LOG(INFO) << "received all weights from clients";

    // convert received vectors to map: class->weight vector
    vector<map<double, vec>> res(n_clients);
    for (int i = 0; i < n_clients; i++) {
        res[i] = arrToMap(X[i], (config->classes).size(), config->rfDim);
    }
    map<double, vec> ans;
    weigthsCombine(res, ans);

    vec y_test;
//    mat x_test = normCol(dataParser->parse(y_test));
    mat x_test = dataParser->parse(y_test);
    double cur_score = validate(ans, Svm::randomFeatures(x_test, config->rfDim, config->rfParam), y_test);

    // free all allocated space
    for (int i = 0; i < n_clients; i++) { delete(X[i]); }
    delete(X);
    delete(rand_w_send);
    delete(bias_send);
}

#endif //SVM_SERVER_H
