#include "Svm.h"
//#include <mpi.h>

#include "../utils/Utils.h"

mat rand_w; // create random w
mat bias;
bool t = false;

map<double, vec> Svm::runLinearSvm(vector<vec> x, vector<double> y, vector<double> classes, double learningRate,
                                   int numIterations, double regParam) {
    map<double, vec> wList;
    //initialize w vector list with zero vectors
    for(unsigned i = 0; i < classes.size(); ++i) {
        wList[classes[i]] = vec(x[0].n_rows).zeros();
    }

    for(int m = 0; m < numIterations; ++m) {
        for(unsigned i = 0; i < x.size(); ++i) {
            // get x_i, y_i and y_hat
            vec x_i = x[i];
            double y_i = y[i];
            double y_hat = argmax(wList, x_i, y_i);

            // check if y_hat is correct
            if((y_hat != y_i) - dot(wList[y_i], x_i) + dot(wList[y_hat], x_i) > 0) {
                // iterate over all classes and update their hypothesis accordingly
                for(unsigned j = 0; j < classes.size(); ++j) {
                    double curClass = classes[j];
                    if(curClass == y_i)
                        wList[curClass] = (1 - learningRate * regParam) * wList[curClass] + learningRate * x_i;
                    else if(curClass == y_hat)
                        wList[curClass] = (1 - learningRate * regParam) * wList[curClass] - learningRate * x_i;
                    else
                        wList[curClass] = (1 - learningRate * regParam) * wList[curClass];
                }
            }
        }
    }
    return wList;
}


//
//
// A sub method of the svm algo.
// The actual update of weights according to it's current prediction.
void Svm::learningUpdate(map<double, vec> &wList, vec classes,
                                double learningRate, double regParam, vec x_i, double y_i, double y_hat) {
    // iterate over all classes and update their hypothesis accordingly
    for(unsigned j = 0; j < classes.size(); ++j) {
        double curClass = classes(j);
        if (curClass == y_i) {
            wList[curClass] = (1 - learningRate * regParam) * wList[curClass] + learningRate * x_i;
        }
        else if (curClass == y_hat) {
            wList[curClass] = (1 - learningRate * regParam) * wList[curClass] - learningRate * x_i;
        }
        else {
            wList[curClass] = (1 - learningRate * regParam) * wList[curClass];
        }
    }
};


//
//
// A sub function of the svm algo.
// Finding argmax and if needing to update, calling the update function.
void Svm::epochLearn(mat x, vec y, map<double, vec> &wList, vec classes,
                            double learningRate, double regParam) {
    for(unsigned i = 0; i < x.n_rows; ++i) {

        vec x_i = x.row(i).t();
        double y_i = y(i);
        double y_hat = argmax(wList, x_i, y_i);
        LOG(DEBUG) << "wList[yhat] cols: " << wList[y_hat].n_cols << " wList[yhat] rows: " << wList[y_hat].n_rows<< " wList[yi] cols: " << wList[y_i].n_cols << " wList[yi] rows: " << wList[y_i].n_rows << " x_i cols " << x_i.n_cols << " x_i rows " << x_i.n_rows << endl;
        LOG(DEBUG) << "y_hat = " << y_hat << endl;
        // check if y_hat is correct
        if((y_hat != y_i) - dot(wList[y_i], x_i) + dot(wList[y_hat], x_i) > 0) {
            learningUpdate(wList, classes, learningRate, regParam, x_i, y_i, y_hat);
        }
    }
};


// Returns the map containing the classes weigths.
//
// The svm Random-Feature algo.
// First calling the randomFeature function for changing it's dimension, than continues
// to the normal svm algo.
map<double, vec> Svm::runKernelRFSvm(mat x, vec y, vec classes, int rfDim, double learningRate,
                                     int numIterations, double regParam, double sigma) {

    x = randomFeatures(x, rfDim, sigma);

    map<double, vec> wList;
    
    //initialize w vector list with zero vectors
    for(unsigned i = 0; i < classes.size(); ++i) {
        wList[classes[i]] = vec(x.n_cols).zeros();
    }

    // Epoch learning
    for(int j = 1; j <= numIterations; ++j) {
        double learningRateT = learningRate / sqrt(j);
        epochLearn(x, y, wList, classes, learningRateT, regParam);
    }
    return wList;
}


// Returns the most correct class id
// calculate y for which w_y will give the max result of w_y_i * x_y + I(y_hat != y_i)
// wList - mapping: class num -> class w (hypotesis vector)
// x_i - current instance vector
//y_i - current classification
//
double Svm::argmax(map<double, vec> wList, vec x_i, double y_i) {
    double maxProduct = -numeric_limits<double>::max();
    double maxY = 0;

    for(map<double, vec>::const_iterator it = wList.begin(); it != wList.end(); ++it) {
        double curClass = it->first;
        vec curW = it->second;
//        cout << "cur class " << curClass << endl;
        double curProduct = dot(curW, x_i) + (y_i != curClass);
//        cout << " dot(curW, x_i) " <<  dot(curW, x_i) << endl;
//        cout << curProduct << endl;
        if(curProduct > maxProduct) {
            maxProduct = curProduct;
            maxY = curClass;
        }
    }
    return maxY;
}


// Returns the new matrix after changing dimension with random feature algo.
// Getting the original matrix - X, the required new dimension and the RF param - sigma.
// Computing the random feauture (once per run - the 't' flag).
// Notice that there is a random element inside this function.

mat Svm::randomFeatures(mat X, int rfDim, double sigma) {
    int NTrain = X.n_rows; // num of instances
    int D = rfDim;
    int NProj = X.n_cols; // num of features

    // run random initialization only once, 't' is a global variable
    if (!t) {
//        LOG(INFO) << "Should not get inside";
        rand_w = mat(D, NProj, fill::randn); // create random w
        rand_w *= sqrt(2.0 * sigma);
        bias = mat(D, 1, fill::randu);
        bias *= 2.0 * M_PI;
        t = true;
    }
    mat G = arma::cos(rand_w * X.t() + repmat(bias, 1, NTrain)) * (sqrt(2) / sqrt(D));

    return G.t();
}


// Returns a double pointer version of the map weights.
// Gets the relevant input for the svm RF run.
// It is an envelope of the runKernelRFSvm function, which just transfer the map
// format to the array format.
double *Svm::runKernelRFSvmWrapper(mat X, int row, int col, vec y, vec classes, int rfDim, double learningRate,
                               int numIterations, double regParam, double sigma) {

    LOG(DEBUG) << "In wrapper SVM, entering real SVM";
    map<double, vec> result = runKernelRFSvm(X, y, classes, rfDim, learningRate, numIterations, regParam, sigma);

    LOG(INFO) << "Done learning, starting to convert map to double*";
    double* res = new double[classes.size() * (rfDim + 1)];
    int i = 0;
    for (map<double, vec>::iterator iter = result.begin(); iter != result.end(); iter++) {

        res[i * (rfDim + 1)] = classes(i);
        for (int j = 1; j <= rfDim; j++) {
            res[i * (rfDim + 1) + j] = result.at(iter->first)[j - 1];
        }
        i++;
    }
    return res;
}




















