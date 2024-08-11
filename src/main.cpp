#include <SFML/Graphics.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <Eigen/Dense>

using Eigen::VectorXd;
using Eigen::MatrixXd;

#include "MLP.hpp"

    // std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    // start = std::chrono::system_clock::now();
    // //
    // //
    // end = std::chrono::system_clock::now();
    
    // std::chrono::duration<double> duration = end - start;
    // std::cout << duration.count() << std::endl;


int main()
{
    std::vector<int> layerInfo = {2, 3, 2};
    MLP mlp(layerInfo);

    std::vector<DataPoint> trainingList;
    // int samples = 4;
    // for(int i = 0; i < samples; i++)
    //     trainingList.emplace_back(DataPoint(layerInfo[0], layerInfo[layerInfo.size() - 1]));
    
    // trainingList[0].input << 0, 0;  trainingList[0].output << 1, 1;
    // trainingList[1].input << 0, 1;  trainingList[1].output << 1, 0; 
    // trainingList[2].input << 1, 0;  trainingList[2].output << 0, 1; 
    // trainingList[3].input << 1, 1;  trainingList[3].output << 0, 0; 


    sf::RenderWindow window(sf::VideoMode(600, 400), "title");
    window.setVerticalSyncEnabled(false);
    sf::Vector2u windowSize = window.getSize();

    sf::Font font;
    font.loadFromFile("rsrc/font.ttf");
    sf::Text epochText;
    sf::Text costText;

    epochText.setFont(font);
    epochText.setPosition(sf::Vector2f(10.f, 10.f));
    
    costText.setFont(font);
    costText.setPosition(sf::Vector2f(10.f, 40.f));


    std::vector<sf::RectangleShape> gridList;
    sf::Vector2f tileSize = {5.f, 5.f};

    gridList.reserve( (windowSize.y/tileSize.y) * (windowSize.x/tileSize.x));

    for(int y = 0; y < windowSize.y/tileSize.y; y++)
        for(int x = 0; x < windowSize.x/tileSize.x; x++)
        {
            gridList.emplace_back(sf::Vector2f(tileSize.x, tileSize.y));
            gridList.back().setPosition(x*tileSize.x, y*tileSize.y);
        }
    
    sf::Color blue = sf::Color(0, 128, 255);
    sf::Color orange = sf::Color(255, 128, 0);

    for(int y = 0; y < windowSize.y/tileSize.y; y++)
        for(int x = 0; x < windowSize.x/tileSize.x; x++)
        {
            VectorXd input(2);
            input << x/(windowSize.x/tileSize.x), y/(windowSize.y/tileSize.y); 
            VectorXd output =  mlp.forwardFeed(input);

            if(output(1) > output(0))
                gridList[y*(windowSize.x/tileSize.x) + x].setFillColor(orange);
            else if(output(1) < output(0))
               gridList[y*(windowSize.x/tileSize.x) + x].setFillColor(blue);
        }

    sf::Event event;
    int pressTimer = 0;
    bool trainingFlag = false;
    int epochs = 0; 
    float cost = 0.f;
    while(window.isOpen())
    {
        while(window.pollEvent(event))
        {
            if(event.type == sf::Event::Closed)
                window.close();
        }
        epochText.setString(std::to_string(epochs));
        costText.setString(std::to_string(cost));

        sf::Vector2f mousePos = (sf::Vector2f)sf::Mouse::getPosition(window);
        sf::Vector2f tileCount = {windowSize.x/tileSize.x, windowSize.y/tileSize.y};
        sf::Vector2f currentTile = {mousePos.x/tileSize.x, mousePos.y/tileSize.y};

        if(sf::Mouse::isButtonPressed(sf::Mouse::Left) && pressTimer == 0)
        { 
            trainingList.emplace_back(DataPoint(layerInfo[0], layerInfo[layerInfo.size() - 1]));
            trainingList.back().input << currentTile.x/tileCount.x, currentTile.y/tileCount.y;
            trainingList.back().output << 1, 0;
            
            trainingList.back().circle.setRadius(tileSize.x);
            trainingList.back().circle.setPosition(mousePos.x - tileSize.x, mousePos.y - tileSize.y);
            trainingList.back().circle.setFillColor(blue);
            trainingList.back().circle.setOutlineThickness(2);
            trainingList.back().circle.setOutlineColor(sf::Color::Black);
            pressTimer = 15;
        }

        if(sf::Mouse::isButtonPressed(sf::Mouse::Right) && pressTimer == 0)
        { 
            trainingList.emplace_back(DataPoint(layerInfo[0], layerInfo[layerInfo.size() - 1]));
            trainingList.back().input << currentTile.x/tileCount.x, currentTile.y/tileCount.y;
            trainingList.back().output << 0, 1;
            
            trainingList.back().circle.setRadius(tileSize.x);
            trainingList.back().circle.setPosition(mousePos.x - tileSize.x, mousePos.y - tileSize.y);
            trainingList.back().circle.setFillColor(orange);
            trainingList.back().circle.setOutlineThickness(2);
            trainingList.back().circle.setOutlineColor(sf::Color::Black);
            pressTimer = 15;
        }

        if(pressTimer > 0)
            pressTimer--;
       
        if(sf::Keyboard::isKeyPressed(sf::Keyboard::Enter) && trainingList.size() != 0 && pressTimer == 0 && trainingFlag == false)
        {
            trainingFlag = true;
            pressTimer = 15;
        }
        if(trainingFlag == true && epochs < 10000)
        {
            for(auto& data : trainingList)
            {
                VectorXd output = mlp.forwardFeed(data.input);
                cost += mlp.cost(data.output, output);
                mlp.backwardFeed(2*(data.output - output), 0.3);
            }
            cost /= trainingList.size();
            for(int y = 0; y < windowSize.y/tileSize.y; y++)
                for(int x = 0; x < windowSize.x/tileSize.x; x++)
                {
                    VectorXd input(2);
                    input << x/(windowSize.x/tileSize.x), y/(windowSize.y/tileSize.y); 
                    VectorXd output =  mlp.forwardFeed(input);

                    if(output(1) > output(0))
                        gridList[y*(windowSize.x/tileSize.x) + x].setFillColor(orange);
                    else if(output(1) < output(0))
                        gridList[y*(windowSize.x/tileSize.x) + x].setFillColor(blue);
                }   
            epochs++;
        }
        if(epochs == 10000)
        {
            trainingFlag = false;
            epochs = 0;
            for(auto& sample : trainingList)
            {
                auto prediction = mlp.forwardFeed(sample.input);
                std::cout << "prediction : " << std::endl; std::cout << prediction << std::endl;
                std::cout << "truth value : " << std::endl; std::cout << sample.output << std::endl;
            }

        }

        window.clear();
        for(auto& grid : gridList)
            window.draw(grid);
        for(auto& data : trainingList)
            window.draw(data.circle);
        window.draw(epochText);
        window.draw(costText);
        window.display();
    }



    return 0;
}