#include "BaseParser.h"

#ifndef SPEECH_RECOGNITION_PROJECT_ADULTPARSER_H
#define SPEECH_RECOGNITION_PROJECT_ADULTPARSER_H

class AdultParser: public BaseParser {
private:
    string instancePath, classificationsPath;
public:
    AdultParser(string instancePath, string classificationsPath);
    mat getInstances();
    vec getClassifications();
};

#endif //SPEECH_RECOGNITION_PROJECT_ADULTPARSER_H
