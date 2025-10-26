#pragma once

class InputNeuron
{
public:
    double value = 0.0;
    
    virtual ~InputNeuron() = default;
    InputNeuron(const double value_); 
};