#include "../Include/InputLayer.hpp"

InputLayer::InputLayer(const InputLayer& source)
{
    neurons.reserve(source.neurons.size());
    neurons = source.neurons;
    /*
    neurons.reserve(source.neurons.size());
    for (std::size_t i = 0; i < neurons.size(); i++)
        neurons[i].value = source.neurons[i].value;
    */
}


InputLayer& InputLayer::operator=(const InputLayer& source)
{
    neurons.reserve(source.neurons.size());
    neurons = source.neurons;

    return *this;
}
