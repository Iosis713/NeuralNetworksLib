#pragma once

#include "InputNeuron.hpp"

#include <vector>

class InputLayer
{
protected:
    std::vector<InputNeuron> inputNeurons;

public:
    friend class NeuralNet;

    InputLayer() = default;
    InputLayer(const std::vector<InputNeuron>& inputNeurons_)
        : inputNeurons(inputNeurons_)
    {};


};
