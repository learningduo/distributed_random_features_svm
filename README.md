Kernel-based Phoneme Distributed Recognizer
===========================================

This repository contains all the work we did as our final project at 3rd year of CS.
We implemented the [Ali Rahimi](https://keysduplicated.com/~ali/) and [Ben Rech](http://www.eecs.berkeley.edu/~brecht/) [article](http://www.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf)
on Random Features which scales-up kernel methods.


## Project Discription
The goal of our project was identifying phonemes from the TIMIT dataset,
 that is, given an unknown phoneme, classifying it as one of 39 possible phonemes
  (reduced set used instead of 44). Therefore, we had to use a classification 
  algorithm for a multi-class problem. We have used the SVM algorithm since 
  it solves a large-margin problem. Using the Random Features kernel of 
  Ali Rahimi and Ben Rech (which their article was mentioned above). 


## Potential Users
Every one who wants to try out the RF kernel algorithm on their data set.
This was written specially for large data sets as Timit, for distributing 
the data on more than one computer and with efficiency consideration.
The most straightforward usage is on the Timit dataset, and [Adult](https://archive.ics.uci.edu/ml/datasets/Adult)
but can be easily converted to any other dataset you might have.

## External libraries
* Using [EasyLogging](https://github.com/easylogging/easyloggingpp) for logging - duh.
* Using [inih](https://github.com/benhoyt/inih) for configuration file parsing
* Using [Catch](https://github.com/philsquared/Catch) for tests.


* Notice that those libraries are included in the project at: *"extern/"* directory

## System Requirements
OS X (10.11.5) and Ubuntu (14.04) are the current supported host development operating systems.

We used CLion in both of them as our IDE.

## Other requirements
To run the application, you will need to have the following:
* [Armadillo](http://arma.sourceforge.net/) for math operation, which it's performance fits best to our needs; at least version 6.600.5 is required
* [open-mpi](https://www.open-mpi.org/software/ompi/v1.10/) which is a library for using distributed machines; at least version 1.10.2 is required
* [openblas](http://www.openblas.net/) -- not required, but **it will** boost your performance; at least version 0.2.17 is required
* [CMake](https://cmake.org/) - we developed this project using CLion, so it was built in (in fact it was the only available generator of build systems); at least version 3.3 is required.
For the difference between the two [read here](http://stackoverflow.com/questions/25789644/makefile-vs-cmake). 

## Environment configuration

* Make sure your dataset is divided into independent chunks. Place these files on the worker (clients) machines.
* Configure the OPEN-MPI hosts file or host parameters according to the FAQ [here](http://open-mpi.org/faq/?category=running).

## Parsing your data

* In order to parse different data sets, one must implement the interface found in "BaseParser.h".
* getInstances should return a mat (matrix representation in armadillo) with your data's instances represented by the rows of the matrix, and features by columns. Same goes for getClassifications, but for the labels of your data.
* Now, just replace the parsing in main() with your new parser class, for example:
```
#!c++
BaseParser *dataParser = new YourNewParser("data-path", "labels-path");
```
* Notice that you can use your parser as you wish, accordingly to your data status. It can be 2 different path, as we did here - for the x and y separately, or as we originally tested on the Mnist dataset, and there you only need one path (as the labels are on the same folder).

or, for a real example:
```
#!c++
BaseParser *dataParser = new TimitParser(config->xTrain, config->yTrain);
```


## Running the project

* You should first configure all the dependent libraries on the CMakeLists.txt file. 
The one given in the project is what is working for us on the computer we finally tested this app.
There shouldn't be much changes to this file.
* In the resources/conf.config you may find settings you may want to tweak:
    * features_dim - the dimensions of the instances (e.g. dimension of "adult" dataset instance is 123).
    * rf_dim - the dimensions of the random features transformation (e.g. in the article we discussed it was 500).
    * n_classes - number of possible classifications (e.g. 2 in the binary case).
    * rf_param - RBF kernel parameter.
    * files - provide with paths to your dataset files.
    * best_param - flag to run hyper-parameter optimization (learning rate and regulatization parameter).
* Locate the location of your build, where your executable file is created.
* Run "mpirun -np <number of machines> <location of executable>"

executable: /home/user/openmpi/bin/mpirun (or wherever you put it)
program params: -np 3 /home/user/path_to_executable

so, for example: /home/yanai/openmpi/bin/mpirun -np 2 /home/yanai/.CLion2016.1/system/cmake/generated/Svm-20557c8e/20557c8e/Debug0/Svm


## Project Structure
### extern
This directory consists external libraries we used for our convenient.
It include 2 libraries:
* logging (easylogging++.h)
* configuration parser (ini.c, ini.h, INIReader.cpp, INIReader.h)

### resources
This directory should consist all of the resources of the project.
 e.g configuration file (conf.config, logging.conf etc...) and the datasets
  which are being used. Notice: the project objective was eventually 
  to make a classifier on the [Timit](https://catalog.ldc.upenn.edu/LDC93S1) 
  which is a private dataset, and a big one (a few gigs.), for these reasons,
   naturally, we couldn't attach it to the project, but anyone who's willing 
   to pay for it, or already have access to it, can verify our final results.
*conf.config*: This file is divided into 4 subsections.
* data: All configurations of running the algorithm via mpi.
There is a possibility here to give explicitly the classes ids, or a range.
Both ones will be parsed and injected to the algorithm.
* files: Where the dataset files sits in the project structure. 
Should be inside the resources folder.
* param_tuning: Consists the parameter tuning section. First of all if to use
this option at all, and if the 'best_param' is set to true, it will run it
on a single machine. The other parameters are the number of time the inned
and outer loop will run.
* final_params: After finding your best parameter, put them inside this section.

### src
All of the code we wrote which is part of the actual project. It's classes are described below:

#### algo
*Svm*: The actual class which implement the svm class. Inside there is the possibility to use the linear solution, or the RF one. The linear method was implemented for benchmark reasons.

#### mpi
*Client*: Inside this class is the job that every client does, as part of the mpi job. Every client reads a part of the data, which should be distributed in advance to each machine. After finishing to calculate the weights, it sends the server it's results.
*Server*: Getting the weights from all clients, and sums them up. Using a simple average for summing up the weights.

#### parsers
*BaseParser*: Implementing here the template design pattern. This class is a pure virtual class, and every true parser which one would use, need to implement 2 methods from this class.
*TimitParser*: is just our parser for parsing the timit data - which also was parsed by the provided python script 
*AdultParser* We also tested our project on this dataset, so we kept it here.

#### utils
*Configuration*: This class stores all the configurations which are being read from the conf.config file. Anything else one would want to add, it should be stored inside.
*Consts*: Just storing const values
*Normalizer*: In the coding process we tried several normalizers, currently using the norma one. One should decide it's best normalizer according to his dataset or perdormance. You are welcome to add here others if you find it usefull.
*Utils* A few static methods we used which didn't had a better place to be.


*main* and finally the main class, which obviously supervise the whole program.


## Tests
We used the *Catch* library for testing our code. All of it can be found in the Tests.cpp file, which contains 
testing method for several classes and function all over the project. The given CMake compiles the tests to a
complete standalone executable which should be run for verifying the project works as expected.
You are welcome to add more tests if you add any feature, or to test a more complicated situations.

#### Written by: Felix Kreuk && Yanai Elazar or should we write Yanai Elazar && Felix Kreuk
##### Supervised by Dr. [Joseph Keshet](http://u.cs.biu.ac.il/~jkeshet/)

For any question, suggestion etc:

felixkreuk@gmail.com

yanaiela@gmail.com