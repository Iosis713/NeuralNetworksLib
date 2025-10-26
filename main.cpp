#include <iostream>

#include "Include/NeuralNet.hpp"

int main()
{
    std::cout << "Hello Neural Networks!\n";
    
    InputLayer inputLayer{{InputNeuron{0.75}, InputNeuron{0.5}}};

    try
    {
        NeuralNet neuralNet{inputLayer, {3, 4}, 2};
        
        std::cout << "\nBefore Forward algorithm:\n";
        neuralNet.Print();
        neuralNet.PrintWeights();

        std::cout << "\nAfter first run of forward algorithm:\n";
        neuralNet.Forward();
        neuralNet.Print();

        std::cout << "\nAfter second run of forward algorithm:\n";
        neuralNet.Forward();
        neuralNet.Print();
    }
    catch(std::runtime_error& err)
    {
        std::cout << err.what() << '\n';
    }

    return 0;
}
