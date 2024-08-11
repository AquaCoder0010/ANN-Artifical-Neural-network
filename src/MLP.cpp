#include "MLP.hpp"

MLP::MLP()
:weightList(), biasList()
{

}
MLP::MLP(std::vector<int>& layerInfo)
{
    dimCount = layerInfo.size() - 1;
    for(int i = 0; i < layerInfo.size(); i++)
    {
        layerList.emplace_back(VectorXd::Constant(layerInfo[i], 0));
        if(i != layerInfo.size() - 1)
        {
            weightList.emplace_back(MatrixXd::Random(layerInfo[i+1], layerInfo[i]));
            biasList.emplace_back(VectorXd::Random(layerInfo[i+1]));
        }
    }
        
}

VectorXd MLP::forwardFeed(const VectorXd& input)
{
    layerList[0] = input;
    for(int i = 0; i < dimCount; i++)
        layerList[i+1] = activation(weightList[i]*layerList[i] + biasList[i]);
    return layerList[layerList.size() - 1];
}


void MLP::trainNetwork(std::vector<DataPoint>& trainingList, float learnRate, int epochs)
{
    for(int i = 0; i < epochs; i++)
        for(auto& data : trainingList)
        {
            auto output = forwardFeed(data.input);
            backwardFeed(2*(data.output - output)/data.output.size(), learnRate);
        }
}


void MLP::backwardFeed(const VectorXd& outputGradient, float learnRate)
{
    VectorXd inputGradient = outputGradient;
    for(int i = dimCount - 1; i >= 0; i--)
    {
        VectorXd modOutput = activationPrime(inputGradient);
        inputGradient = weightList[i].transpose()*modOutput;

        weightList[i] += learnRate*(modOutput*layerList[i].transpose());
        biasList[i] += learnRate*(modOutput);
    }
}

//
VectorXd MLP::activation(const VectorXd& input)
{
    return tanh(input.array());
}

VectorXd MLP::activationPrime(const VectorXd& outputGradient)
{
    VectorXd inputGradient = outputGradient.unaryExpr([&](double x){ return 1 - tanh(x)*tanh(x); });
    inputGradient.array() *= outputGradient.array();
    return inputGradient;
}
/*
    VectorXd xPrime = outputGradient.unaryExpr([&](double x){ return 1 - tanh(x)*tanh(x); });
    VectorXd deltaX(xPrime.size());
    for(int i = 0; i < xPrime.size(); i++)
        deltaX(i) = outputGradient(i)*xPrime(i);
    return deltaX;
*/


