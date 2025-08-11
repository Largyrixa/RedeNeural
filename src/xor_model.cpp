#include "rede_neural.h"

#include <cstdlib>

int main()
{
    nn::Sequencial rede({2, 2, 2}, "SCE");

    nn::Matriz entradas = { {0, 0}, {0, 1}, {1, 0}, {1, 1} };
    nn::Matriz saidas   = { {0, 1}, {1, 0}, {1, 0}, {0, 1} };

    rede.train(entradas, saidas, entradas, saidas, 1e-3, 1000, 0.0001, 0);

    rede.salvar_rede("data/xor_model.txt");

    for (int i = 0; i < 4; i++)
    {
        printf("ENTRADA: {%.0lf, %.0lf}\n", entradas[i][0], entradas[i][1]);
        printf("ESPERADO: {%.2lf, %.2lf}\n", saidas[i][0], saidas[i][1]);
        
        auto previsao = rede.feed_forward(entradas[i]);

        printf("PREVISTO: {%.2lf, %.2lf}\n\n", previsao[0], previsao[1]);
    }
}