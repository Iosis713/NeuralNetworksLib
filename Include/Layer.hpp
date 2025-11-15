#pragma once

#include <vector>
#include "Neuron.hpp"

class Layer
{
protected:
    std::vector<Neuron> neurons;

public:
friend class NeuralNet;
    Layer() = default;
    Layer(const std::size_t layerSize, const std::size_t previousLayerSize);

    std::vector<Neuron>& GetNeurons() { return this->neurons; }

};
