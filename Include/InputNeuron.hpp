#pragma once

class InputNeuron
{
protected:
    

public:
    double value = 0.0;
    
    virtual ~InputNeuron();
    InputNeuron(const double value_);
    double inline GetValue() { return this->value; }
    
};