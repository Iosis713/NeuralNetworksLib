#pragma once

#include "InputLayer.hpp"
#include "Layer.hpp"

#include <functional>
#include <cmath>
#include <concepts>
#include <type_traits>

namespace ActivationFunction
{
    void Sigmoid(double& value);//Logistic
    void TanH(double& value);
};

namespace ActivationFunctionDerivative
{
    inline double Sigmoid(const double value) { return value * (1.0 - value); }
    inline double TanH(const double value) { return 1.0 - std::pow(value, 2); }
}

template<typename T>
concept LayerT = requires(T t)
{
    requires std::same_as<decltype(t.neurons), std::vector<typename decltype(t.neurons)::value_type>>;
    requires std::derived_from<typename decltype(t.neurons)::value_type, InputNeuron>;
};

class NeuralNet
{
private:
    InputLayer inputLayer;
    std::vector<Layer> hiddenLayers;
    Layer outputLayer;
    std::function<void(double&)> activationFunction = ActivationFunction::TanH;

    template<LayerT PreviousLayer>
    void ComputeLayer(Layer& layer, const PreviousLayer& previousLayer)
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


public:
    //temporarly constructor with random weights initialization, as an input to genetic algorithm
    //constructor reading values from file will be added soon
    NeuralNet(const InputLayer& inputLayer_
            , const std::vector<std::size_t>& hiddenLayersSizes
            , const std::size_t outputLayerSize);

    void Forward();
    void BackPropagation(const std::vector<double>& expected, const double learningRate);
    void Print();
    void PrintWeights();

};
