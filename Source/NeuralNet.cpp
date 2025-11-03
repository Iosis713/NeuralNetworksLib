#include "../Include/NeuralNet.hpp"

#include <stdexcept>
#include <iostream>
#include <format>

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
            hiddenLayers.emplace_back(Layer{ hiddenLayersSizes[0], inputLayer.neurons.size()});
        else
            hiddenLayers.emplace_back(Layer{ hiddenLayersSizes[hiddenLayerSizeIndex], hiddenLayersSizes[hiddenLayerSizeIndex -1 ]});
    }

    outputLayer = Layer{ outputLayerSize, hiddenLayers.at(hiddenLayers.size() - 1).neurons.size() };
}

void NeuralNet::ComputeInputLayer()
{
    for (auto& neuron : hiddenLayers.at(0).neurons)
    {
        neuron.value = 0.0;
        auto previousLayerNeuron = inputLayer.neurons.begin();
        
        for (auto weight : neuron.weights)
        {
            neuron.value += weight * previousLayerNeuron->value;
            previousLayerNeuron++;
        }

        activationFunction(neuron.value);
    }
}

void NeuralNet::ComputeLayer(Layer& layer, const Layer& previousLayer)
{
    for (auto& neuron : layer.neurons)
    {
        neuron.value = 0.0;
        auto previousLayerNeuron = previousLayer.neurons.begin();
        
        for (auto weight : neuron.weights)
        {
            neuron.value += weight * previousLayerNeuron->value;
            previousLayerNeuron++;
        }

        activationFunction(neuron.value);
    }   
}

void NeuralNet::Forward()
{
    ComputeInputLayer();

    for (std::size_t hiddenLayerIndex = 1; hiddenLayerIndex < hiddenLayers.size(); hiddenLayerIndex++)
        ComputeLayer(hiddenLayers[hiddenLayerIndex], hiddenLayers[hiddenLayerIndex - 1]);

    ComputeLayer(outputLayer, hiddenLayers.back());
}

void NeuralNet::Print()
{
    std::cout << "FORMAT:\n"
              << "Input Layer:\n"
              << "Hidden Layers [0...n]:\n"
              << "Output Layer: \n";
    
    for (const auto& neuron : inputLayer.neurons)
        std::cout << neuron.value << " "; 

    std::cout << '\n';

    for (const auto& layer : hiddenLayers)
    {
        for (const auto& neuron : layer.neurons)
            std::cout << neuron.value << " "; 
            
        std::cout << '\n';
    }

    for (const auto& neuron : outputLayer.neurons)
        std::cout << neuron.value << " "; 

    std::cout << '\n';
}

void NeuralNet::PrintWeights()
{
    std::cout << "Weights:\n";
    for (const auto& layer : hiddenLayers)
    {
        for (const auto& neuron : layer.neurons)
        {
            for (const auto weight : neuron.weights)
            {
                std::cout << weight << " ";
            }
            std::cout << " || ";
        }
        std::cout << '\n';
    }
    std::cout << '\n';
}


namespace ActivationFunction
{
    void Sigmoid(double& value)//Logistic
    {
        value = 1.0 / (1.0 + std::exp(-value));
    };

    void TanH(double& value)
    {
        value = std::tanh(value);
    }   
}