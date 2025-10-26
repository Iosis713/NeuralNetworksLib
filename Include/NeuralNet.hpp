#pragma once

#include "InputLayer.hpp"
#include "Layer.hpp"

#include <functional>
#include <cmath>

namespace ActivationFunction
{
    void Sigmoid(double& value);//Logistic
    void TanH(double& value);
};

class NeuralNet
{
private:
    InputLayer inputLayer;
    std::vector<Layer> hiddenLayers;
    Layer outputLayer;
    std::function<void(double&)> activationFunction = ActivationFunction::TanH;

    void ComputeInputLayer();
    void ComputeLayer(Layer& layer, const Layer& previousLayer);


public:
    //temporarly constructor with random weights initialization, as an input to genetic algorithm
    //constructor reading values from file will be added soon
    NeuralNet(const InputLayer& inputLayer_
            , const std::vector<std::size_t>& hiddenLayersSizes
            , const std::size_t outputLayerSize);

    
    void Forward();
    void Print();
    void PrintWeights();

};
