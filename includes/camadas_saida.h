#ifndef _CAMADAS_SAIDA_H
#define _CAMADAS_SAIDA_H

#include <vector>
#include <string>
#include <cmath>
#include <numeric>
#include <memory>

namespace 
{
    using Vetor = std::vector<double>;
}

// Interface para todas as combinações de Ativação/Loss da camada de saída
class CamadaSaida
{
public:
    virtual ~CamadaSaida() = default;

    // Calcula a saída ativada apartir dos logits (somas ponderadas)
    virtual Vetor forward(const Vetor &logits) = 0;

    // Calcula o gradiente inicial (delta) para o backpropagation
    virtual Vetor backward(const Vetor &saida_ativada, const Vetor &saida_esperada) = 0;

    // Calcula o valor de loss para monitoramento
    virtual double calcular_loss(const Vetor &saida_ativada, const Vetor &saida_esperada) = 0;

    virtual std::string get_tipo() const = 0;

    virtual std::unique_ptr<CamadaSaida> clone() const = 0;
};

// --- ESTRATÉGIA PARA CLASSIFICAÇÃO ---
class SoftmaxCrossEntropy : public CamadaSaida
{
public:
    Vetor forward(const Vetor &logits) override
    {
        Vetor ativacoes = logits;
        if (ativacoes.empty())
            return ativacoes;

        double max_val = ativacoes[0];

        for (size_t i = 1; i < ativacoes.size(); i++)
            max_val = std::max(max_val, ativacoes[i]);

        double sum = 0.0;
        for (size_t i = 0; i < ativacoes.size(); i++)
        {
            ativacoes[i] = exp(ativacoes[i] - max_val);
            sum += ativacoes[i];
        }

        for (size_t i = 0; i < ativacoes.size(); i++)
            ativacoes[i] /= sum;

        return ativacoes;
    }

    Vetor backward(const Vetor &saida_ativada, const Vetor &saida_esperada) override
    {
        Vetor delta = saida_ativada;
        for (size_t i = 0; i < delta.size(); i++)
        {
            delta[i] -= saida_esperada[i];
        }
        return delta;
    }

    double calcular_loss(const Vetor &saida_ativada, const Vetor &saida_esperada) override
    {
        // Perda de Entropia Cruzada Categórica
        double perda = 0.0;
        for (size_t i = 0; i < saida_esperada.size(); i++)
        {
            // Adiciona um valor pequeno para evitar log(0)
            perda += saida_esperada[i] * log(saida_ativada[i] + 1e-9);
        }
        return -perda;
    }

    std::string get_tipo() const override { return "SCE"; }

    std::unique_ptr<CamadaSaida> clone() const override
    {
        return std::make_unique<SoftmaxCrossEntropy>(*this);
    }

}; // SoftmaxCrossEntropy

// --- ESTRATÉGIA PARA REGRESSÃO ---
class LinearMeanSquareError : public CamadaSaida
{
public:
    Vetor forward(const Vetor &logits) override
    {
        // A ativação linear simplesmente retorna a entrada
        return logits;
    }

    Vetor backward(const Vetor &saida_ativada, const Vetor &saida_esperada) override
    {
        // Derivada do Erro Quadrático Médio: (saída - esperado)
        // Multiplicado pela derivada da ativação linear (1)
        Vetor delta = saida_ativada;
        for (size_t i = 0; i < delta.size(); i++)
        {
            delta[i] -= saida_esperada[i];
        }
        return delta;
    }

    double calcular_loss(const Vetor &saida_ativada, const Vetor &saida_esperada) override
    {
        // Erro Quadrático Médio (Mean Squared Error)
        double perda = 0.0;
        for (size_t i = 0; i < saida_esperada.size(); i++)
        {
            perda += pow(saida_ativada[i] - saida_esperada[i], 2);
        }
        return perda / saida_ativada.size();
    }

    std::string get_tipo() const override { return "LMSE"; }

    std::unique_ptr<CamadaSaida> clone() const override
    {
        return std::make_unique<LinearMeanSquareError>(*this);
    }

}; // LinearMeanSquareError

#endif // _CAMADAS_SAIDA_H