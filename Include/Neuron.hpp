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

    Neuron(const std::size_t inputSize_);



};
