#pragma once

#include "InputLayer.hpp"
#include "Layer.hpp"

class NeuralNet
{
protected:
    InputLayer inputLayer;
    std::vector<Layer> hiddenLayers;
    Layer outputLayer;

public:
    NeuralNet(const InputLayer& inputLayer_
            , const std::vector<std::size_t>& hiddenLayersSizes
            , const std::size_t outputLayerSize);

};
