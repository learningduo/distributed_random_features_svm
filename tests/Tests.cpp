#include "../src/algo/Svm.h"
#include "../src/parsers/TimitParser.h"
#include "../src/parsers/AdultParser.h"
#include "../src/utils/Normalizer.h"
#include "../src/mpi/Server.h"
#include "../extern/catch.hpp"
#include "../extern/easylogging++.h"
INITIALIZE_EASYLOGGINGPP
extern bool t;


void init() {
    // Logging init.
// Load configuration from file
    el::Configurations conf("/home/yanai/workspace/Svm/resources/logging.conf");
// Reconfigure single logger
    el::Loggers::reconfigureLogger("default", conf);
// Actually reconfigure all loggers instead
    el::Loggers::reconfigureAllLoggers(conf);
// Now all the loggers will use configuration from file

    el::Loggers::addFlag(el::LoggingFlag::ColoredTerminalOutput);
}


mat createMat() {
    mat x(3, 4);
    for (int i = 0; i < 4; i++) {
        x(0, i) = 0;
        x(1, i) = 1;
    }
    for (int i = 0; i < 4; i++) {
        x(2, i) = i;
    }
    return x;
}

double* createArr() {
    double *res = new double[10];
    res[0] = -1;
    for (int i = 1; i < 5; i++) {
        res[i] = i;
    }
    res[5] = 1;
    for (int i = 6; i < 10; i++) {
        res[i] = i;
    }
    return res;
}

TEST_CASE("SVM", "svm") {
    init();


    SECTION("Svm argmax") {
        map<double, vec> weights;
        weights[1] = vec(5).zeros();
        weights[2] = vec(5).ones();
        vec x {1, 2, 3, 4, 5};
        double ans = Svm::argmax(weights, x, 1);
        REQUIRE(ans == 2);
        weights[3] = vec(5).ones() * 2;
        ans = Svm::argmax(weights, x, 1);
        REQUIRE(ans == 3);
    }

    SECTION("Svm algo") {
        BaseParser *dataParser = new AdultParser("/home/yanai/workspace/Svm/resources/adult/a_train.csv", "/home/yanai/workspace/Svm/resources/adult/a_ytrain.csv");
        BaseParser *testParser = new AdultParser("/home/yanai/workspace/Svm/resources/adult/a_test.csv", "/home/yanai/workspace/Svm/resources/adult/a_ytest.csv");

        vec classes {-1, 1};
        vec y;
        mat x = normCol(dataParser->parse(y));

        // For running a single learning
        Svm svm;
        map<double, vec> result = svm.runKernelRFSvm(x, y, classes, 500, 57.5, 5, 0.00001, 19.0);

        vec y_test;
        mat x_test = normCol(testParser->parse(y_test));
        double cur_score = validate(result, Svm::randomFeatures(x_test, 500, 19.0), y_test);
        REQUIRE(cur_score >= 0.6);

    }

    SECTION("Random Feature") {

        t = false;
        mat x = mat(10, 2, fill::randn);
        mat X = Svm::randomFeatures(x, 10, 1.0);

        REQUIRE(x.n_rows == 10);
        REQUIRE(x.n_cols == 2);
        REQUIRE(X.n_rows == 10);
        REQUIRE(X.n_cols == 10);

    }

}


TEST_CASE("Parser", "parse") {

    SECTION("Adult Parser") {
        BaseParser *dataParser = new AdultParser("/home/yanai/workspace/Svm/resources/adult/a_train.csv", "/home/yanai/workspace/Svm/resources/adult/a_ytrain.csv");
        BaseParser *testParser = new AdultParser("/home/yanai/workspace/Svm/resources/adult/a_test.csv", "/home/yanai/workspace/Svm/resources/adult/a_ytest.csv");

        vec y;
        mat x = dataParser->parse(y);

        REQUIRE(x.n_rows > 30000);
        REQUIRE(x.n_cols == 123);

        mat xTest = testParser->parse(y);

        REQUIRE(xTest.n_rows > 10000);
        REQUIRE(xTest.n_cols == 123);
    }
}

TEST_CASE("Normalizers", "normalize") {

    SECTION("Normalize by Column") {
        mat x = createMat();

        mat xNorm = normCol(x);

        for (int i = 0; i < 4; i++) {
            REQUIRE(xNorm(0, i) == 0);
            REQUIRE(xNorm(1, i) <= 1);
            REQUIRE(xNorm(1, i) >= 0);
            REQUIRE(xNorm(2, i) <= 1);
            REQUIRE(xNorm(2, i) >= 0);
        }


    }
}


TEST_CASE("Utils") {
    SECTION("arrToMap") {
        double *res = createArr();

        map<double, vec> ans = arrToMap(res, 2, 4);
        REQUIRE(ans.size() == 2);
        vec v1 = ans.at(-1);
        for (int i = 0; i < 4; i++) {
            REQUIRE(v1[i] == i + 1);
        }

        vec v2 = ans.at(1);
        for (int i = 0; i < 4; i++) {
            REQUIRE(v2[i] == i + 6);
        }
        delete res;
    }

    SECTION("arrFromMat") {
        mat x = createMat();
        double *arr = arrFromMat(x);
        REQUIRE(arr[0] == 0);
        REQUIRE(arr[1] == 0);
        REQUIRE(arr[2] == 0);
        REQUIRE(arr[3] == 0);
        REQUIRE(arr[4] == 1);
        REQUIRE(arr[5] == 1);
        REQUIRE(arr[6] == 1);
        REQUIRE(arr[7] == 1);
        REQUIRE(arr[8] == 0);
        REQUIRE(arr[9] == 1);
        REQUIRE(arr[10] == 2);
        REQUIRE(arr[11] == 3);

        delete arr;
    }

    SECTION("matFromArr") {
        double *arr = createArr();
        mat x = matFromArr(arr, 2, 5);
        REQUIRE(x(0, 0) == -1);
        REQUIRE(x(0, 1) == 1);
        REQUIRE(x(0, 2) == 2);
        REQUIRE(x(0, 3) == 3);
        REQUIRE(x(0, 4) == 4);

        REQUIRE(x(1, 0) == 1);
        REQUIRE(x(1, 1) == 6);
        REQUIRE(x(1, 2) == 7);
        REQUIRE(x(1, 3) == 8);
        REQUIRE(x(1, 4) == 9);

        delete arr;
    }
}