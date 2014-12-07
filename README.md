# Erlang-Shen

Erlang-Shen is a naive concurrent implementation of the neural network learning algorithm in Erlang by [Alex Lenail](https://github.com/zfrenchee) and [Sunjay Bhatia](https://github.com/sunjayBhatia) for COMP 50-02 Concurrent Programming at Tufts University. It is named after the Chinese truth-seeing god, [Erlang Shen](https://en.wikipedia.org/wiki/Erlang_Shen). Through this implementation, we hope to use the power of concurrency in Erlang to take a different approach to neural networks and model them in a way that is more analagous to how the brain functions. Our network currently can only handle binary class predictions, but we hope to expand on this in the future.


## Usage

#### Download
        git clone https://github.com/zfrenchee/Erlang-Shen.git
        cd Erlang-Shen

#### Build:
        make clean && make

#### Start Erlang/OTP:
        erl -pa ebin

#### Start and run application:
        application:start(shen).
        shen:run(TrainingDataFile, TestDataFile, HiddenLayerDimensions, GradientDescentSteps).
Where `TrainingDataFile` and `TestDataFile` are strings representing paths to valid ARFF format data files. Example files can be found in the `datasets` folder of this repository. `HiddenLayerDimensions` is a list of integers that can be specified as the hidden layer architecture of the neural network. `GradientDescentSteps` is an integer specifiying the number of gradient descent steps to take to tune the network. Results are displayed and output to the `results` folder in this repository.

#### Stop application:
        application:stop(shen).
