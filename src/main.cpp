#include <iostream>
#include <armadillo>
#include <map>
#include <sys/stat.h>
#include <mpi.h>

#include "algo/Svm.h"
#include "parsers/BaseParser.h"
#include "parsers/AdultParser.h"
#include "parsers/TimitParser.h"
#include "utils/Normalizer.h"
#include "utils/Configuration.h"
#include "mpi/Client.h"
#include "mpi/Server.h"

#include "../extern/cmdline.h"
#include "../extern/INIReader.h"

#include "utils/Consts.h"


INITIALIZE_EASYLOGGINGPP


extern mat rand_w;
extern mat bias;
extern bool t;


using namespace std;
using namespace arma;
using namespace cmdline;



// Returns the configuration parser.
// Initializing the logger and the input parser.
// This is the place to declare more configurable input options.
INIReader initConfig() {

    // Logging init.
    // Load configuration from file
    el::Configurations conf(LOG_FILE);
    // Reconfigure single logger
    el::Loggers::reconfigureLogger("default", conf);
    // Actually reconfigure all loggers instead
    el::Loggers::reconfigureAllLoggers(conf);
    // Now all the loggers will use configuration from file

    el::Loggers::addFlag(el::LoggingFlag::ColoredTerminalOutput);

//    LOG(DEBUG);
    INIReader reader(CONFIG_FILE);
    if (reader.ParseError() < 0) {
        LOG(ERROR) << "Can't load the 'conf.config'";
        exit(-1);
    }

    string xTrain = reader.Get(FILES_SECTION, TRAIN, "resources/adult/a_train.csv");
    string yTrain = reader.Get(FILES_SECTION, LABEL, "resources/adult/a_ytrain.csv");

    LOG(INFO) << xTrain;
    LOG(INFO) << yTrain;
    // Checking if the data files exists
    struct stat buffer;
    if (stat (xTrain.c_str(), &buffer) != 0 || stat (yTrain.c_str(), &buffer) != 0) {
        LOG(ERROR) << "One of the files does not exists...";
        exit(-1);
    }

    return reader;
}


// Getting the relevant data for running the svm.
// Running svm with several parameters, for finding the optimized ones.
void findBestParams(mat x, mat x_test, vec y, vec y_test, Configuration *config) {
    double learn_max = 0.0;
    double reg_max = 0.0;
    double simga_max = 0.0;
    double best_score = -1.0;

    const double tuningSigStart = config->tuningSigStart;
    const double tuningSigJump = config->tuningSigJump;
    const double tuningLearnStart = config->tuningLearnStart;
    const double tuningLearnJump = config->tuningLearnJump;

    for (int i = 0; i < config->outer; i++) {
        for (int j = 0; j < config->inner; j++) {
            // Learning
            Svm *svm = new Svm();
            double sigma = tuningSigStart + tuningSigJump * j;
            double learn = tuningLearnStart + tuningLearnJump * i;
            double reg = 0.00001;
            LOG(DEBUG) << "learning rate: " << learn << " reg param: " << reg << " sigma: " << sigma << endl;
            map<double, vec> svm_result = arrToMap(svm->runKernelRFSvmWrapper(x, x.n_rows, x.n_cols, y,
                                                                               config->classes, config->rfDim, learn,
                                                                               config->iterations, reg, sigma),
                                                   config->classes.size(), config->rfDim);

            // Validation
            double cur_score = validate(svm_result, svm->randomFeatures(x_test, config->rfDim, sigma), y_test);
            if (cur_score > best_score) {
                learn_max = learn;
                reg_max = reg;
                simga_max = sigma;
                best_score = cur_score;

                LOG(INFO) << "learn_max: " << learn_max;
                LOG(INFO) << "reg_max: " << reg_max;
                LOG(INFO) << "sigma: " << simga_max;
                LOG(INFO) << "best_score: " << best_score;
            }

            delete(svm);
            t = false;
        }
    }

    LOG(INFO) << "Results: " << "learn param: " << learn_max << " reg param: "
    << reg_max << " sigma param: " << simga_max << " with best score of: " << best_score;
}


void callingBestParam(BaseParser *dataParser, BaseParser *testParser, Configuration *config) {
    vec y;
    vec y_test;
    mat x = dataParser->parse(y);
    mat x_test = testParser->parse(y_test);
//    mat x_and_test = join_cols(x, x_test);
//    x_and_test = normCol(x_and_test);
//    x = x_and_test.submat(0, 0, x.n_rows-1, x_and_test.n_cols-1);
//    x_test = x_and_test.submat(x.n_rows, 0, x_and_test.n_rows-1, x_and_test.n_cols-1);
    findBestParams(x, x_test, y, y_test, config);
}


// Getting the main input.
// Making all the initialization of the program.
// Calling consequently the relevant execution:
// * running a single process for finding best params
// * running svm on several machines (or processes) for single solution with fixed params
int main(int argc, char* argv[]) {

    MPI_Init(NULL, NULL);
    // Find out rank, size
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    INIReader reader = initConfig();
    Configuration *config = new Configuration(reader);

    // --- GET CONFIGURATION ---
    BaseParser *dataParser = new TimitParser(config->xTrain, config->yTrain);
//    BaseParser *dataParser = new AdultParser(config->xTrain, config->yTrain);
    BaseParser *testParser = new TimitParser(config->xTest, config->yTest);
//    BaseParser *testParser = new AdultParser(config->xTest, config->yTest);


    clock_t begin_time;

    // --- RUN SERVER+CLIENTS or TUNING ROUTINE ---
    if (config->bestParam) {
        LOG(INFO) << "Looking for best parameters...";
        if (world_rank == 0) {
            callingBestParam(dataParser, testParser, config);
        }
    } else {
        LOG(INFO) << "Making normal learning...";
        if (world_rank == 0) {
            begin_time = clock();
            mpiServer(testParser, config);

        } else {
            mpiClients(dataParser, config);
        }
    }

    LOG(INFO) << "time from start: " << float(clock () - begin_time) / CLOCKS_PER_SEC;

    delete(config);
    delete(dataParser);
    delete(testParser);

    MPI_Finalize();

    return 0;
}
