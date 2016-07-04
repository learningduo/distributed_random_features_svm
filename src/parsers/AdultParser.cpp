#include "AdultParser.h"

AdultParser::AdultParser(string instancePath, string classificationsPath) {
    this->instancePath = instancePath;
    this->classificationsPath = classificationsPath;
}

mat AdultParser::getInstances() {
    ifstream infile(this->instancePath);
    string line;
    vector<vec> instances;
    while (getline(infile, line))
    {
        vector<string> splitLine = split(line, ',');
        vec curVec(123);
        for(int i=0;i<splitLine.size(); i++){
            curVec(i) = stof(splitLine[i]);
        }
        instances.push_back(curVec);
    }
    mat X(instances.size(), instances[0].n_rows);
    for (int i = 0; i < instances.size(); i++) {
        for (int j = 0; j < instances[0].n_rows; j++) {
            X(i, j) = instances[i][j];
        }
    }
    return X;
}

vec AdultParser::getClassifications() {
    string line;
    ifstream infile2(this->classificationsPath);
    vector<double> y_temp;
    while (std::getline(infile2, line))
    {
        y_temp.push_back(stof(line));
    }
    return vec(y_temp);
}