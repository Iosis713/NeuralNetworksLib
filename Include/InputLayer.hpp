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

    InputLayer(const InputLayer& source);

    std::vector<InputNeuron>& GetNeurons() { return this->neurons; };
    InputLayer& operator=(const InputLayer& source);

};
