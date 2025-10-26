#include "../Include/Neuron.hpp"

#include <random>

Neuron::Neuron(const std::size_t inputSize_)
    : InputNeuron(RandomInitialise())
    , inputSize(inputSize_)
{
    weights.reserve(inputSize);
    for (std::size_t weightIndex = 0; weightIndex < inputSize; weightIndex++)
    {
        weights.push_back(RandomInitialise());
    }
}

double Neuron::RandomInitialise()
{
    std::mt19937 randomGenerator{ std::random_device{}() };
    std::uniform_real_distribution<double> distribution{ -1.0, 1.0 };
    return distribution(randomGenerator);
}
