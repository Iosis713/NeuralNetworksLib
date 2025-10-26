#pragma once

#include "InputNeuron.hpp"

#include <vector>

class InputLayer
{
protected:
    std::vector<InputNeuron> inputNeurons;

public:
    friend class NeuralNet;


};
