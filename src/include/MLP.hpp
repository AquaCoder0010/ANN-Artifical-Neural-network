#pragma once
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <SFML/Graphics/CircleShape.hpp>

using Eigen::VectorXd;
using Eigen::MatrixXd;

static double tanhPrime(double x)
{
    return 1 - tanh(x)*tanh(x);
}

struct DataPoint
{
    VectorXd input;
    VectorXd output;
    sf::CircleShape circle;
    DataPoint(int inputSize, int outputSize)
    :input(), output()
    {
        input = VectorXd::Constant(inputSize, 0);
        output = VectorXd::Constant(outputSize, 0);
    }
};

class MLP
{
    private:
        std::vector<MatrixXd> weightList;
        std::vector<VectorXd> layerList;
        std::vector<VectorXd> biasList;
        int dimCount = 0; 
    public:
        MLP();
        MLP(std::vector<int>& layerInfo);
        VectorXd forwardFeed(const VectorXd& input);
        void backwardFeed(const VectorXd& outputGradient, float learnRate);
        void trainNetwork(std::vector<DataPoint>& trainingList, float learnRate, int epochs);
    private:
        VectorXd activation(const VectorXd& input);
        VectorXd activationPrime(const VectorXd& outputGradient);
    public:
        float cost(const VectorXd& desiredOutput, const VectorXd& output)
        {
            return (desiredOutput - output).array().pow(2).mean();
        }

};