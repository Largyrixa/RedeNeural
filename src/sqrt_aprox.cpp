#include "rede_neural.h"

#include <cstdlib>
#include <cmath>
#include <random>
#include <vector>

using namespace std;

int main()
{
    nn::Sequencial rede({1, 10, 1}, "LMSE");

    random_device rd;
    mt19937 gen(rd());  
    uniform_int_distribution<int> num(0, 100000);
    uniform_int_distribution<int>     which(1, 10);

    nn::Matriz entradas, saidas, entradas_2, saidas_2;

    for (int i = 0; i < 10000; i++)
    {
        double x = num(gen);
        double y = sqrt(abs(x));

        nn::Vetor saida;
        if (x < 0)
        {
            saida = {y};
        }
        else
        {
            saida = {y};
        }

        if (which(gen) == 1)
        {
            entradas_2.push_back({x});
            saidas_2.push_back(saida);
        }
        else
        {
            entradas.push_back({x});
            saidas.push_back(saida);
        }
    }

    for (int i = 0; i < 100; i++)
    {
        entradas_2.push_back({(double)i});
        saidas_2.push_back({sqrt(i)});
    }

    rede.train(entradas, saidas, entradas_2, saidas_2, 0.0001, 100, 0.01, 1e-5);

    rede.salvar_rede("data/sqrt_aprox_model.txt");
    
    for (int i = 0; i < 10; i++)
    {
        double x = i;
        double y = sqrt(abs(x));

        nn::Vetor saida = rede.feed_forward({x});

        printf("PREVISÃO: sqrt(%.2lf) = %.2lf", x, saida[0]);
        //if (saida[1] > 0.5) printf("i");
        printf("\n");

        printf("VERDADE: sqrt(%.2lf) = %.2lf", x, y);
        //if (x < 0) printf("i");
        printf("\n\n");
    }
}