#include <iostream>

#include "Include/NeuralNet.hpp"

int main()
{
    std::cout << "Hello Neural Networks!\n";
    
    InputLayer inputLayer{{InputNeuron{0.75}, InputNeuron{0.5}}};

    try
    {
        //NeuralNet neuralNet{inputLayer, {3, 4}, 1};
        /*
        std::cout << "\nBefore Forward algorithm:\n";
        neuralNet.Print();
        neuralNet.PrintWeights();

        std::cout << "\nAfter first run of forward algorithm:\n";
        neuralNet.Forward();
        neuralNet.Print();

        std::cout << "\nAfter second run of forward algorithm:\n";
        neuralNet.Forward();
        neuralNet.Print();
        */
        /*
        std::cout << "\nFirst backpropagation run:\n";
        neuralNet.BackPropagation();
        neuralNet.Print();
        neuralNet.PrintWeights();
        */


        /*
        std::vector<std::vector<double>> inputs = {
    {0, 0},
    {0, 1},
    {1, 0},
    {1, 1}
};

std::vector<std::vector<double>> outputs = {
    {0}, {1}, {1}, {0}
};

NeuralNet net(inputLayer, {2}, 1); // 2 hidden neurons, 1 output

double lr = 0.1;
for (int epoch = 0; epoch < 10000; ++epoch) {
    for (size_t i = 0; i < inputs.size(); ++i) {
        for (size_t j = 0; j < inputs[i].size(); ++j)
            net.inputLayer.neurons[j].value = inputs[i][j];

        net.Forward();
        net.BackPropagation(outputs[i], lr);
    }
}

// Test results
for (size_t i = 0; i < inputs.size(); ++i) {
    for (size_t j = 0; j < inputs[i].size(); ++j)
        net.inputLayer.neurons[j].value = inputs[i][j];

    net.Forward();
    std::cout << "Input: " << inputs[i][0] << ", " << inputs[i][1]
              << " -> Output: " << net.outputLayer.neurons[0].value << "\n";
}*/
        
        std::vector<InputLayer> inputs = {
            {{InputNeuron{0}, InputNeuron{0}}},
            {{InputNeuron{0}, InputNeuron{1}}},
            {{InputNeuron{1}, InputNeuron{0}}},
            {{InputNeuron{1}, InputNeuron{1}}}
        };

        std::vector<std::vector<double>> outputs = {
            {-1}, {1}, {1}, {-1}
        };

        NeuralNet net(inputLayer, {4}, 1); // 2 hidden neurons, 1 output
        
        double learningRate = 0.05;
        for (int epoch = 0; epoch < 100000; ++epoch) {
            for (size_t i = 0; i < inputs.size(); ++i) {
                for (size_t j = 0; j < inputs[i].GetNeurons().size(); ++j)
                    net.GetInputLayer().GetNeurons()[j].value = inputs[i].GetNeurons()[j].value;

                net.Forward();
                net.BackPropagation(outputs[i], learningRate);
            }
        }

        // Test results
        for (size_t i = 0; i < inputs.size(); ++i) {
            for (size_t j = 0; j < inputs[i].GetNeurons().size(); ++j)
                net.GetInputLayer().GetNeurons()[j].value = inputs[i].GetNeurons()[j].value;

            net.Forward();
            std::cout << "Input: " << inputs[i].GetNeurons()[0].value << ", " << inputs[i].GetNeurons()[1].value
                      << " -> Output: " << net.GetOutputLayer().GetNeurons()[0].value << "\n";
        }
    }
    catch(std::runtime_error& err)
    {
        std::cout << err.what() << '\n';
    }

    return 0;
}
