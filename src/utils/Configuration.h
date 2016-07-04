#include <vector>
#include <armadillo>
#include "../../extern/INIReader.h"
#include "../../extern/easylogging++.h"
#include "Consts.h"

#ifndef SVM_CONFIGURATION_H
#define SVM_CONFIGURATION_H

using namespace std;
using namespace arma;

class Configuration {
public:

    int xDim, rfDim;
    double rfParam;
    vec classes;

    string xTrain, yTrain, xTest, yTest;

    bool bestParam;
    int outer, inner;

    double learningRate, reg;
    int iterations;

    double tuningSigStart, tuningLearnStart, tuningSigJump, tuningLearnJump;

    Configuration(INIReader &reader) {

        this->tuningSigStart = reader.GetReal(PARAMS_TUNING_SECTION, "sig_start", 0.1);
        this->tuningLearnStart = reader.GetReal(PARAMS_TUNING_SECTION, "learn_start", 10.0);
        this->tuningSigJump = reader.GetReal(PARAMS_TUNING_SECTION, "sig_jump", 0.1);
        this->tuningLearnJump = reader.GetReal(PARAMS_TUNING_SECTION, "learn_jump", 0.1);

        this->xDim = reader.GetInteger(DATA_SECTION, "features_dim", 39);;
        this->rfDim = reader.GetInteger(DATA_SECTION, "rf_dim", 500);
        this->rfParam = reader.GetReal(DATA_SECTION, "rf_param", 19.0);

        string classesRange = reader.Get(DATA_SECTION, "classes_range", "0-39");
        string classesExplicit = reader.Get(DATA_SECTION, "classes_explicit", "NULL");
        getRange(classesRange, classesExplicit);


        this->bestParam = reader.GetBoolean(PARAMS_TUNING_SECTION, "best_param", false);
        this->outer = reader.GetInteger(PARAMS_TUNING_SECTION, "outer", 2);
        this->inner = reader.GetInteger(PARAMS_TUNING_SECTION, "inner", 2);

        this->xTrain = reader.Get(FILES_SECTION, "x_train", "resources/adult/a_train.csv");
        this->yTrain = reader.Get(FILES_SECTION, "y_train", "resources/adult/a_ytrain.csv");
        this->xTest = reader.Get(FILES_SECTION, "x_test", "resources/adult/a_test.csv");
        this->yTest = reader.Get(FILES_SECTION, "y_test", "resources/adult/a_ytest.csv");

        this->learningRate = reader.GetReal(FINAL_PARAMS_SECTION, LEARNING_RATE, 57.5);
        this->reg = reader.GetReal(FINAL_PARAMS_SECTION, REGULARIZATION, 0.00001);
        this->iterations = reader.GetInteger(FINAL_PARAMS_SECTION, ITERATIONS, 5);

        LOG(INFO) << "configs: \nx_train=" << this->xTrain
        << "\ny_train=" << this->yTrain
        << "\nfeatures-dim=" << this->xDim
        << "\nRandom features-dim=" << this->rfDim
        << "\nNumber of classes=" << reader.GetInteger("data", "n_classes", 2)
        << "\nRun best params=" << reader.GetBoolean("misc", "best_param", false)
        << "\nRandom features sigma=" << this->rfParam
        << "\nEnd of config file\n";
    }

    void getRange(string range, string explicitVec) {
        vector<int> parsedRange = vector<int>(2);
        string delimiter = "-";
        size_t pos = 0;
        string token;

        vector<double> finalRange;
        if (explicitVec.compare("NULL") == 0) {
            LOG(INFO) << "explicit = null";
            parsedRange.at(0) = stoi(range.substr(0, range.find(delimiter)));
            range.erase(0, pos + delimiter.length() + 1);
            parsedRange.at(1) = stoi(range);

            vector<int> parsedExclude;
            for (int i = parsedRange.at(0); i <= parsedRange.at(1); i++) {
                finalRange.push_back(i);
            }
        } else {
            LOG(INFO) << "explicit != null";
            string delimiter = ",";
            while ((pos = explicitVec.find(delimiter)) != std::string::npos) {
                token = explicitVec.substr(0, pos);
                finalRange.push_back(stoi(token));
                explicitVec.erase(0, pos + delimiter.length());
            }
            finalRange.push_back(stoi(explicitVec));
        }
        this->classes = vec(finalRange);

    }
};

#endif //SVM_CONFIGURATION_H
