#include <iostream>
#include <cmath>

using namespace std;

double inputValue;

double Function(double x);
void Learn(double learnRate);

int main(void)
{
    inputValue = -3;
    const double learningRate = 0.05;
    const int epochs = 1000;
    for (int i = 0; i <= epochs; i++)
    {
        Learn(learningRate);
        cout << inputValue;
        cout << "\n";
    }
}

double Function(double x)
{
    return 0.2 * pow(x, 4) + 0.1 * pow(x, 3) - pow(x, 2) + 2;
}

void Learn(double learnRate)
{
    const double h = 0.00001;
    double deltaOutput = Function(inputValue + h) - Function(inputValue);
    double slope = deltaOutput / h;

    inputValue -= slope * learnRate;
}
