#include "BaseParser.h"

mat BaseParser::parse(vec &classifications) {
    classifications = getClassifications();
    return getInstances();
}

vector<string> &BaseParser::split(const string &s, char delim, vector<string> &elems) {
    stringstream ss(s);
    string item;
    while (getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}

vector<string> BaseParser::split(const string &s, char delim) {
    vector<string> elems;
    split(s, delim, elems);
    return elems;
}


/*
mat Parser::parseMnist(string xPath, vec &y) {
    ifstream infile(xPath);
    string line;
    vector<vec> instances;
    vector<double> y_temp;
    while (getline(infile, line))
    {
        vector<string> splitLine = split(line, ',');
        y_temp.push_back(stoi(splitLine[0]));
        vec curVec(784);
        for(int i = 1; i < splitLine.size(); i++){
            curVec(i - 1) = stoi(splitLine[i]);
        }
        instances.push_back(curVec);
    }

    mat X(instances.size(), instances[0].n_rows);

    for (int i = 0; i < instances.size(); i++) {
        for (int j = 0; j < instances[0].n_rows; j++) {
            X(i, j) = instances[i][j];
        }
    }

    LOG(INFO) << "finished reading Mnist file";
    y  = vec(y_temp);

    return X;
}
 */