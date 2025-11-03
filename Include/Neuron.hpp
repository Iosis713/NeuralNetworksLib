#pragma once

#include "InputNeuron.hpp"

#include <vector>

class Neuron : public InputNeuron
{
private:
    std::size_t inputSize;

    double RandomInitialise();
    
public:
    std::vector<double> weights;
    double error = 0.0;

    Neuron(const std::size_t inputSize_);
};
