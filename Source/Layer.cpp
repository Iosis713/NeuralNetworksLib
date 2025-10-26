#include "../Include/Layer.hpp"

Layer::Layer(const std::size_t layerSize, const std::size_t previousLayerSize)
{
    neurons.reserve(layerSize);
    for (std::size_t i = 0; i < layerSize; i++)
        neurons.push_back(Neuron{previousLayerSize});

}
