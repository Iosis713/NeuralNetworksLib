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

void NeuralNet::Forward()
{
    ComputeLayer(hiddenLayers[0], inputLayer);

    for (std::size_t hiddenLayerIndex = 1; hiddenLayerIndex < hiddenLayers.size(); hiddenLayerIndex++)
        ComputeLayer(hiddenLayers[hiddenLayerIndex], hiddenLayers[hiddenLayerIndex - 1]);

    ComputeLayer(outputLayer, hiddenLayers.back());
}

void NeuralNet::BackPropagation(const std::vector<double>& expected, const double learningRate)
{
    //Output layers errors
    for (std::size_t i = 0; i < outputLayer.neurons.size(); i++)
    {
        auto& neuron = outputLayer.neurons[i];
        double output = neuron.value;
        neuron.error = (expected[i] - output) * ActivationFunctionDerivative::Sigmoid(output);;
    }

    //Hidden layers
    for (int layerIndex = (hiddenLayers.size() - 1); layerIndex >= 0; --layerIndex)
    {
        auto& layer = hiddenLayers[layerIndex];
        const Layer& nextLayer = (layerIndex == static_cast<int>(hiddenLayers.size() - 1)) ? outputLayer : hiddenLayers[layerIndex + 1];

        for (std::size_t i = 0; i < layer.neurons.size(); i++)
        {
            auto& neuron = layer.neurons[i];
            double sum = 0.0;

            for (auto& nextNeuron : nextLayer.neurons)
                sum += nextNeuron.weights[i] * nextNeuron.error;

            neuron.error = ActivationFunctionDerivative::Sigmoid(neuron.value) * sum;
        }
    }

    //Update weights
    //Hidden -> Output

    {
        const Layer& previousLayer = hiddenLayers.back();
        for (auto& neuron : outputLayer.neurons)
        {
            for (std::size_t weightIndex = 0; weightIndex < neuron.weights.size(); weightIndex++)
                neuron.weights[weightIndex] += learningRate * neuron.error * previousLayer.neurons[weightIndex].value;
        }
    }

    //Input -> hidden layers
    for (std::size_t layerIndex = 0; layerIndex < hiddenLayers.size(); layerIndex++)
    {
        if (layerIndex == 0)
        {
            auto& neuron = hiddenLayers[layerIndex].neurons.at(0);
            for (std::size_t w = 0; w < neuron.weights.size(); w++)
                neuron.weights[w] += learningRate * neuron.error * inputLayer.neurons[w].value;
        }
        else
        {
            const auto& previousLayer = hiddenLayers[layerIndex - 1];
            for (auto& neuron : hiddenLayers[layerIndex].neurons)
            {
                for (std::size_t w = 0; w < neuron.weights.size(); w++)
                    neuron.weights[w] += learningRate * neuron.error * previousLayer.neurons[w].value;
            }
        }
    }
}

void NeuralNet::Train(const std::vector<InputLayer>& trainingInputs, const std::vector<std::vector<double>>& expected, const double learningRate, const long epochs)
{
    for (long epoch = 0; epoch < epochs; epoch++)
    {
        for (size_t i = 0; i < trainingInputs.size(); i++)
        {
            for (size_t j = 0; j < trainingInputs[i].neurons.size(); j++)
                inputLayer.neurons[j].value = trainingInputs[i].neurons[j].value;

            Forward();
            BackPropagation(expected[i], learningRate);
        }
    }
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