#include <iostream>

#include "Include/NeuralNet.hpp"

int main()
{
    std::cout << "Hello Neural Networks!\n";
    
    InputLayer inputLayer{{InputNeuron{0.75}, InputNeuron{0.5}}};

    try
    {
        // TEST with XOR 
        std::vector<InputLayer> inputs = {
            {{InputNeuron{0}, InputNeuron{0}}},
            {{InputNeuron{0}, InputNeuron{1}}},
            {{InputNeuron{1}, InputNeuron{0}}},
            {{InputNeuron{1}, InputNeuron{1}}}
        };

        std::vector<std::vector<double>> outputs = {
            {0}, {1}, {1}, {0}
        };

        NeuralNet net(inputLayer, {6, 6}, 1);
        constexpr double learningRate = 0.1;
        constexpr long maxEpochs = 100'000;
        net.Train(inputs, outputs, learningRate, maxEpochs);

        // Test results
        for (size_t i = 0; i < inputs.size(); ++i) {
            for (size_t j = 0; j < inputs[i].GetNeurons().size(); ++j)
                net.GetInputLayer() = inputs[i];
            
            net.Forward();
            net.Print();
        }

        net.PrintWeights();
    }
    catch(std::runtime_error& err)
    {
        std::cout << err.what() << '\n';
    }

    return 0;
}
