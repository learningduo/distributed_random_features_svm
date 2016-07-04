#ifndef SVM_HTKPARSER_H
#define SVM_HTKPARSER_H

#include "BaseParser.h"

class TimitParser : public BaseParser {
private:
    string instancePath, classificationsPath;
public:
    TimitParser(string instancePath, string classificationsPath);
    mat getInstances();
    vec getClassifications();
};


#endif //SVM_HTKPARSER_H
