#pragma once

#include "InputNeuron.hpp"

#include <vector>

class InputLayer
{
protected:
    std::vector<InputNeuron> neurons;

public:
    friend class NeuralNet;

    InputLayer() = default;
    InputLayer(const std::vector<InputNeuron>& neurons_)
        : neurons(neurons_)
    {};


};
