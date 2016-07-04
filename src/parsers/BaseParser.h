#ifndef SPEECH_RECOGNITION_PROJECT_BASEPARSER_H
#define SPEECH_RECOGNITION_PROJECT_BASEPARSER_H
#include <armadillo>
using namespace std;
using namespace arma;

class BaseParser {
protected:
    static vector<string> &split(const string &s, char delim, vector<string> &elems);
    static vector<string> split(const string &s, char delim);
public:
    /**
     * generic parsing function, returns a mat object representing the instances,
     * and assigns the label vector to classifications.
     */
    mat parse(vec &classifications);
    /**
     * both of the following functions are to be implemented by the user. getInstances()
     * should return the mat object representing the instances, and getClassifications()
     * should return the vec with label. see AdultParser for example.
     */
    virtual mat getInstances() = 0;
    virtual vec getClassifications() = 0;


};

#endif //SPEECH_RECOGNITION_PROJECT_BASEPARSER_H
