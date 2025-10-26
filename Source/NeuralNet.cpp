#include "../Include/NeuralNet.hpp"

#include <stdexcept>

NeuralNet::NeuralNet(const InputLayer& inputLayer_
                    , const std::vector<std::size_t>& hiddenLayersSizes
                    , const std::size_t outputLayerSize)
    : inputLayer(inputLayer_)
{
    hiddenLayers.reserve(hiddenLayersSizes.size());

    if (!hiddenLayersSizes.size())
        throw std::runtime_error("Invalid number of hidden layers. At least one is required");

    for (std::size_t hiddenLayerSizeIndex = 0; hiddenLayerSizeIndex < hiddenLayersSizes.size(); hiddenLayerSizeIndex++)
    {
        if (hiddenLayerSizeIndex == 0)
            hiddenLayers.emplace_back(Layer{ hiddenLayersSizes[0], inputLayer.inputNeurons.size()});
        else
            hiddenLayers.emplace_back(Layer{ hiddenLayersSizes[hiddenLayerSizeIndex], hiddenLayersSizes[hiddenLayerSizeIndex -1 ]});
    }

    outputLayer = Layer{ outputLayerSize, hiddenLayers.at(hiddenLayers.size() - 1).neurons.size() };
}