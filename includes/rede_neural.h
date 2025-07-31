#ifndef _REDE_NEURAL_H
#define _REDE_NEURAL_H

#include <vector>
#include <string>

class RedeNeural {
public:
    // Construtor: define a arquitetura
    // (ex: 24 entradas, 16 neurônios ocultos, 4 saídas)

    RedeNeural (int inputs, int hiddens, int outputs);

    // Função principal: recebe as entradas e retorna as saídas
    std::vector<double> feedFoward(const std::vector<double>& entradas) const;

    // Métodos para o algorítmo genético
    
    // Pesos entre a camada de entrada e a camada oculta
    std::vector<std::vector<double>> pesos_ih;
    // Pesos entre a camada oculta e a camada de saída
    std::vector<std::vector<double>> pesos_ho;

    // Bias para os neurônios da camada oculta e de saída
    std::vector<double> bias_h;
    std::vector<double> bias_o;

    // Número de neurônios de entrada
    int inputs;
    // Número de neurônios ocultos
    int hiddens;
    // Número de neurônios de saída
    int outputs;

    RedeNeural& operator=(const RedeNeural& other);

    bool carregar_rede(const std::string& path);
    bool salvar_rede(const std::string& path) const;
};

#endif