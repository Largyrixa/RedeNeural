#include "rede_neural.h"
#include "camadas_saida.h"

#include <vector>
#include <string>
#include <functional>
#include <random>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <exception>
#include <cmath>
#include <utility>
#include <iostream>
#include <deque>
#include <omp.h>
#include <iterator>

using namespace nn;

namespace nn
{
    std::unique_ptr<CamadaSaida> camada_saida_padrao = std::make_unique<LinearMeanSquareError>();
    func ativ_oculta_padrao = {ReLU, dReLU};
}

//
// CONSTRUTORES
//

Sequencial::Sequencial(
    const std::vector<size_t> &topologia,
    std::string camada_saida_str,
    func funcao_ativacao_oculta) : m_topologia(topologia),
                                   funcao_ativacao_oculta(funcao_ativacao_oculta),
                                   m_timestep(0)
{
    // Alocação de espaço nos vetores
    m_pesos.resize(m_topologia.size() - 1);
    m_pesos_m.resize(m_topologia.size() - 1);
    m_pesos_v.resize(m_topologia.size() - 1);
    m_gradientes_pesos.resize(m_topologia.size() - 1);

    m_biases.resize(m_topologia.size() - 1);
    m_biases_m.resize(m_topologia.size() - 1);
    m_biases_v.resize(m_topologia.size() - 1);
    m_gradientes_biases.resize(m_topologia.size() - 1);

    for (int i = 0; i < m_topologia.size() - 1; i++)
    {
        int neuronios_atual = m_topologia[i];
        int neuronios_prox = m_topologia[i + 1];

        m_pesos[i].resize(neuronios_atual);
        m_pesos_m[i].resize(neuronios_atual);
        m_pesos_v[i].resize(neuronios_atual);
        m_gradientes_pesos[i].resize(neuronios_atual);
        for (int j = 0; j < neuronios_atual; j++)
        {
            m_pesos[i][j].resize(neuronios_prox);
            m_pesos_m[i][j].resize(neuronios_prox, 0.0);
            m_pesos_v[i][j].resize(neuronios_prox, 0.0);
            m_gradientes_pesos[i][j].resize(neuronios_prox);
        }

        m_biases[i].resize(neuronios_prox);
        m_biases_m[i].resize(neuronios_prox, 0.0);
        m_biases_v[i].resize(neuronios_prox, 0.0);
        m_gradientes_biases[i].resize(neuronios_prox);
    }

    if (camada_saida_str == "SCE")
    {
        m_camada_saida = std::make_unique<SoftmaxCrossEntropy>();
    }
    else // tipo padrão
    {
        m_camada_saida = std::make_unique<LinearMeanSquareError>();
    }

    inicializar_pesos();
    inicializar_biases();
} // Sequencial

Sequencial::Sequencial(const std::string &caminho, func funcao_ativacao_oculta) : m_camada_saida(std::move(camada_saida_padrao))
{
    if (!carregar_rede(caminho, funcao_ativacao_oculta))
    {
        throw std::invalid_argument("arquivo com sintaxe inválida!");
    }
}

void Sequencial::inicializar_pesos()
{
    std::random_device rd;
    std::mt19937 generator(rd());

    for (int i = 0; i < m_pesos.size(); i++)
    {
        std::uniform_real_distribution<double> dist(0.0, std::sqrt(2.0 / m_topologia[i]));

        for (int j = 0; j < m_pesos[i].size(); j++)
            for (int k = 0; k < m_pesos[i][j].size(); k++)
            {
                double peso_aleatorio = dist(generator);
                m_pesos[i][j][k] = peso_aleatorio;
            }
    }
}

void Sequencial::inicializar_biases()
{
    for (int i = 0; i < m_biases.size(); i++)
        for (int j = 0; j < m_biases[i].size(); j++)
        {
            m_biases[i][j] = 0.0;
        }
}

//
// LÓGICA DA REDE
//

Vetor Sequencial::feed_forward(const Vetor &entradas) const
{
    // Verifica se a entrada tem o tamanho correto
    if (entradas.size() != m_topologia.front())
    {
        return {};
    }

    // Limpa os caches da chamada anterior
    m_logits.clear();
    m_ativacoes.clear();

    m_ativacoes.push_back(entradas); // A ativação da camada 0 são as próprias entradas

    Vetor camada_atual_valores = entradas;

    // --- Entrada -> Ocultas -> Saída ---

    // Itera através de cada camada de conexão (pesos e biases)
    // O índice 'i' representa a conexão entre a camada 'i' e 'i+1'
    for (size_t i = 0; i < m_pesos.size(); ++i)
    {
        Vetor proxima_camada_logits(m_topologia[i + 1]);

        // Calcula a soma ponderada para cada neurônio da próxima camada (logits)
        #pragma omp parallel for 
        for (size_t j = 0; j < proxima_camada_logits.size(); ++j)
        {
            double soma_ponderada = 0.0;
            for (size_t k = 0; k < camada_atual_valores.size(); ++k)
            {
                // Acesso aos pesos:
                // m_pesos[camada][neuronio_origem][neuronio_destino]
                soma_ponderada += camada_atual_valores[k] * m_pesos[i][k][j];
            }

            // Adiciona o bias
            soma_ponderada += m_biases[i][j];
            proxima_camada_logits[j] = soma_ponderada;
        }

        m_logits.push_back(proxima_camada_logits); // Guarda os logits da camada

        // Aplica a ativação
        Vetor proxima_camada_ativacoes(m_topologia[i + 1]);
        if (i < m_pesos.size() - 1) // Camadas ocultas
        {
            #pragma omp parallel for
            for (size_t j = 0; j < proxima_camada_logits.size(); j++)
            {
                proxima_camada_ativacoes[j] = funcao_ativacao_oculta.funcao(proxima_camada_logits[j]);
            }
            camada_atual_valores = proxima_camada_ativacoes;
            m_ativacoes.push_back(camada_atual_valores);
        }
    }

    // Aplica a ativação de fato (os logits de saída)
    Vetor saida_final = m_camada_saida->forward(m_logits.back()); // Usa os últimos logits calculados
    m_ativacoes.push_back(saida_final);                           // Atualiza o último elemento das ativações com o resultado final

    return saida_final;
} // feed_forward

// APRENDIZADO DE MÁQUINA
void Sequencial::backpropagate(const Vetor &saida_esperada)
{
    /*
    Obs: observe que, ao contrário do feed_forward, neste método estamos
    começando da última camada (de saída) para poder calcular o gradiente
    de erro de cada neurônio.
    */

    //=======================================================//
    //  PASSO 1: Calcular o erro (delta) da CAMADA DE SAÍDA  //
    //=======================================================//
    Vetor ultima_ativacao = m_ativacoes.back();
    Vetor delta = m_camada_saida->backward(ultima_ativacao, saida_esperada);

    // Agora, com o delta da camada de saída, calculamos os gradientes para os
    // pesos e biases que se conectam a ela (a última camada de pesos/biases)

    // Índice da última camada de pesos/biases
    size_t L = m_pesos.size() - 1;

    // A ativação da penúltima camada (que alimenta a camada de saída)
    Vetor ativacao_camada_anterior = m_ativacoes[L];

    // Calcular gradientes para a última camada de conexões
    // Para cada neurônio 'j' na camada de SAÍDA:
    #pragma omp parallel for
    for (size_t j = 0; j < m_topologia.back(); j++)
    {
        m_gradientes_biases[L][j] = delta[j];

        // Para cada neurônio 'k' na camada ANTERIOR:
        for (size_t k = 0; k < m_topologia[L]; k++)
        {
            // O gradiente do peso é o sinal de erro do neurônio de destino
            // multiplicado pela ativação do neurônio de origem.
            m_gradientes_pesos[L][k][j] = ativacao_camada_anterior[k] * delta[j];
        }
    }

    //====================================================//
    //  PASSO 2: Propagar o erro para as CAMADAS OCULTAS  //
    //====================================================//
    // Agora, iteramos para trás, da penúltima camada até a primeira.

    // O índice 'L' representa a camada de CONEXÕES (pesos/biases).
    for (long L = m_pesos.size() - 2; L >= 0; L--)
    {
        const Vetor &delta_camada_seguinte = delta; // 'delta' da iteração anterior
        const Matriz &pesos_camada_seguinte = m_pesos[L + 1];
        const Vetor &logits_camada_atual = m_logits[L];

        Vetor novo_delta(m_topologia[L + 1]); // O novo delta para a camada (L+1)

        // Para cada neurônio 'k' na camada atual (L+1)
        #pragma omp parallel for
        for (size_t k = 0; k < m_topologia[L + 1]; k++)
        {
            double erro_propagado = 0.0;
            // Somar o erro vindo de cada neurônio 'j' da camada seguinte (L+2)
            for (size_t j = 0; j < m_topologia[L + 2]; j++)
            {
                erro_propagado += delta_camada_seguinte[j] * pesos_camada_seguinte[k][j];
            }

            double derivada_ativacao = funcao_ativacao_oculta.derivada(logits_camada_atual[k]);
            novo_delta[k] = erro_propagado * derivada_ativacao;
        }

        delta = novo_delta;

        // Agora, com o novo delta, calculamos os gradientes para a camada de pesos L
        const Vetor &ativacao_camada_anterior = m_ativacoes[L];
        #pragma omp parallel for
        for (size_t j = 0; j < m_topologia[L + 1]; j++)
        {
            m_gradientes_biases[L][j] = delta[j];
            for (size_t k = 0; k < m_topologia[L]; k++)
            {
                m_gradientes_pesos[L][k][j] = ativacao_camada_anterior[k] * delta[j];
            }
        }
    }
} // backpropagate

void Sequencial::otimizar(double taxa_aprendizagem, double beta1 = 0.9,
                          double beta2 = 0.999, double epsilon = 1e-8)
{
    double gradiente, atualizacao;
    double m, v;         // momentos
    double m_hat, v_hat; // biases dos momentos

    // Incrementa o contador de tempo (para correção de bias)
    m_timestep++;
    long t = m_timestep;

    //===============================//
    //  PASSO 1: Atualizar os pesos  //
    //===============================//
    // Para cada camada de pesos L
    #pragma omp parallel for
    for (size_t L = 0; L < m_pesos.size(); L++)
        // Para cada neurônio de origem k
        #pragma omp parallel for
        for (size_t k = 0; k < m_pesos[L].size(); k++)
            // Para cada neurônio de destino j
            #pragma omp parallel for
            for (size_t j = 0; j < m_pesos[L][k].size(); j++)
            {
                // Pega o gradiente calculado pelo backpropagate
                gradiente = m_gradientes_pesos[L][k][j];

                // 1. Atualiza o primeiro momento (média dos gradientes)
                m = m_pesos_m[L][k][j];
                m_pesos_m[L][k][j] = beta1 * m + (1.0 - beta1) * gradiente;

                // 2. atualiza o segundo momento (média dos gradientes ao quadrado)
                v = m_pesos_v[L][k][j];
                m_pesos_v[L][k][j] = beta2 * v + (1.0 - beta2) * pow(gradiente, 2.0);

                // 3. Corrige o bias dos momentos
                m_hat = m_pesos_m[L][k][j] / (1.0 - pow(beta1, t));
                v_hat = m_pesos_v[L][k][j] / (1.0 - pow(beta2, t));

                // 4. Calcula a atualização do peso
                // epsilon é usado para evitar divisão por zero
                atualizacao = taxa_aprendizagem * m_hat / (sqrt(v_hat) + epsilon);

                // 5. Aplica a atualização
                m_pesos[L][k][j] -= atualizacao;
            }

    //================================//
    //  PASSO 2: Atualizar os biases  //
    //================================//
    // Para cada camada de pesos L
    #pragma omp parallel for
    for (size_t L = 0; L < m_biases.size(); L++)
        // Para cada bias j
        #pragma omp parallel for
        for (size_t j = 0; j < m_biases[L].size(); j++)
        {
            // Pega o gradiente calculado pelo backpropagate
            gradiente = m_gradientes_biases[L][j];

            // 1. Atualiza o primeiro momento
            m = m_biases_m[L][j];
            m_biases_m[L][j] = beta1 * m + (1.0 - beta1) * gradiente;

            // 2. Atualiza o segundo momento
            v = m_biases_v[L][j];
            m_biases_v[L][j] = beta2 * v + (1.0 - beta2) * pow(gradiente, 2.0);

            // 3. Corrige o bias dos momentos
            m_hat = m_biases_m[L][j] / (1.0 - pow(beta1, t));
            v_hat = m_biases_v[L][j] / (1.0 - pow(beta2, t));

            // 4. Calcula a atualização do peso
            atualizacao = taxa_aprendizagem * m_hat / (sqrt(v_hat) + epsilon);

            // 5. Aplica a atualização
            m_biases[L][j] -= atualizacao;
        }
} // otimizar

double Sequencial::calc_loss(const std::vector<Vetor> &entradas, const std::vector<Vetor> &saidas_esperadas) const
{
    double perda_total = 0.0;
    for (size_t i = 0; i < entradas.size(); i++)
    {
        auto saida_ativada = feed_forward(entradas[i]);
        perda_total += m_camada_saida->calcular_loss(saida_ativada, saidas_esperadas[i]);
    }

    perda_total /= entradas.size();
    return perda_total;
}

namespace
{
    size_t argmax(const nn::Vetor& vec)
    {
        return std::distance(vec.begin(), std::max_element(vec.begin(), vec.end()));
    }
}

double Sequencial::calc_accuracy(const std::vector<Vetor> &entradas, const std::vector<Vetor> &saidas_esperadas) const
{
    if (entradas.empty() || entradas.size() != saidas_esperadas.size())
    {
        return 0.0;
    }

    int acertos = 0;

    #pragma omp parallel for reduction(+:acertos)
    for (size_t i = 0; i < entradas.size(); ++i)
    {
        Vetor previsao = feed_forward(entradas[i]);

        if (previsao.empty()) continue;

        size_t index_previsto = argmax(previsao);
        size_t index_real     = argmax(saidas_esperadas[i]);

        if (index_previsto == index_real)
        {
            acertos++;
        }
    }

    return static_cast<double>(acertos) / entradas.size();
}

void Sequencial::train(const std::vector<Vetor> &entradas_treino, const std::vector<Vetor> &saidas_treino,
                       const std::vector<Vetor> &entradas_validacao, const std::vector<Vetor> &saidas_validacao,
                       double taxa_aprendizagem, size_t janela_analise, double target_loss, double threshold)
{
    if (janela_analise < 2) janela_analise = 2;
    double melhor_perda = INFINITY;

    std::vector<Matriz> melhores_pesos;
    std::vector<Vetor>  melhores_biases;

    std::deque<double> historico_loss;

    for (size_t epoca = 1; epoca > 0; epoca ++)
    {
        for (size_t i = 0; i < entradas_treino.size(); i++)
        {
            feed_forward(entradas_treino[i]); // para gerar os logits
            backpropagate(saidas_treino[i]);
            otimizar(taxa_aprendizagem);
        }

        auto perda_atual = calc_loss(entradas_validacao, saidas_validacao);
        
        historico_loss.push_front(perda_atual);
        
        long double desvio_padrao = 0.0;
        if (historico_loss.size() > janela_analise)
        {
            historico_loss.pop_back();

            long double media = 0.0;
            for (double loss : historico_loss)
            {
                media += loss;
            }
    
            media /= historico_loss.size();
    
            for (double loss : historico_loss)
            {
                desvio_padrao += pow((loss - media), 2);
            }
    
            desvio_padrao = sqrt(desvio_padrao / (double)historico_loss.size());
    
            
            if (desvio_padrao <= threshold)
            {
                std::cout << ">>> LOSS ESTABILIZADO <<<\n";
                std::cout << "TREINAMENTO FINALIZADO NA ÉPOCA " << epoca << std::endl;
                break;
            }
            
        }
        
        if (epoca % 1 == 0)
        {
            auto precisao = calc_accuracy(entradas_validacao, saidas_validacao);
            std::cout << "ÉPOCA: " << epoca <<
            "\nLOSS: "<< perda_atual << 
            "\nPRECISÃO: " << precisao * 100.0 << "% (SCE)"<<
            "\nDP: " << desvio_padrao << std::endl << std::endl;
        }

        if (perda_atual < melhor_perda)
        {
            melhores_pesos = m_pesos;
            melhores_biases= m_biases;
            melhor_perda = perda_atual;
        }

        if (melhor_perda >= 0 && melhor_perda <= target_loss)
        {
            std::cout << ">>> ALVO ATINGIDO <<<\n";
            std::cout << "TREINAMENTO FINALIZADO NA ÉPOCA " << epoca << std::endl;
            break;
        }

    }
    

    std::cout << ">>> FIM DO TREINO <<< " << std::endl << std::endl;
    std::cout << "LOSS FINAL: " << melhor_perda << std::endl;

    m_pesos = melhores_pesos;
    m_biases = melhores_biases;
} // train

//
// SETTERS
//

void Sequencial::set_pesos(int index_camada, const Matriz &novos_pesos)
{
    if (index_camada < m_pesos.size() && novos_pesos.size() == m_pesos[index_camada].size())
    {
        for (int i = 0; i < m_pesos[index_camada].size(); ++i)
        {
            if (novos_pesos[i].size() != m_pesos[index_camada][i].size())
            {
                throw std::runtime_error("O index da camada é inválido");
            }
        }
        m_pesos[index_camada] = novos_pesos;
    }
}

void Sequencial::set_biases(int index_camada, const Vetor &novos_biases)
{
    if (index_camada < m_biases.size() && novos_biases.size() == m_biases[index_camada].size())
    {
        m_biases[index_camada] = novos_biases;
    }
    else
    {
        throw std::runtime_error("O index da camada é inválido");
    }
}

void Sequencial::set_func(func ativ_oculta, std::unique_ptr<CamadaSaida> camada_saida)
{
    funcao_ativacao_oculta = ativ_oculta;
    m_camada_saida = std::move(camada_saida);
}

//
// GETTERS
//

const Matriz &Sequencial::get_pesos(int index_camada) const
{
    if (index_camada < 0 || index_camada >= m_topologia.size() - 1)
    {
        throw std::runtime_error("O index da camada é inválido");
    }

    return m_pesos[index_camada];
}

const Vetor &Sequencial::get_biases(int index_camada) const
{
    if (index_camada < 1 || index_camada >= m_topologia.size())
    {
        throw std::runtime_error("O index da camada é inválido");
    }

    return m_biases[index_camada - 1];
}

const std::vector<size_t> &Sequencial::get_topologia() const
{
    return m_topologia;
}

//
// MÉTODOS DE PERSISTÊNCIA
//

bool Sequencial::salvar_rede(const std::string &caminho) const
{
    std::ofstream file(caminho, std::fstream::out | std::fstream::trunc);

    file << "#######################################" << std::endl;
    file << "# ARQUIVO DE DESCRIÇÃO DE REDE NEURAL #" << std::endl;
    file << "# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- #" << std::endl;
    file << "#       >>> REDE SEQUENCIAL <<<       #" << std::endl;
    file << "#######################################" << std::endl;

    file << std::endl;

    file << "##########################" << std::endl;
    file << "# definição da topologia #" << std::endl;
    file << "##########################" << std::endl;

    file << std::endl;

    file << "#" << std::endl;
    file << "# CAMADA index número_de_entradas" << std::endl;
    file << "#" << std::endl;

    file << std::endl;

    for (size_t i = 0; i < m_topologia.size(); i++)
    {
        file << "CAMADA " << i << " " << m_topologia[i] << std::endl;
    }

    file << std::endl;

    file << "#" << std::endl;
    file << "# TIPO_SAIDA TipoDeCamadaSaida" << std::endl;
    file << "#" << std::endl;
    file << std::endl;

    file << "TIPO_SAIDA " << m_camada_saida->get_tipo() << std::endl;
    file << std::endl;

    file << "#######################" << std::endl;
    file << "# definição de biases #" << std::endl;
    file << "#######################" << std::endl;
    file << std::endl;

    file << "#" << std::endl;
    file << "# BIAS camada bias_0 bias_1 ... bias_n-1" << std::endl;
    file << "#" << std::endl;
    file << std::endl;

    for (size_t i = 0; i <= m_biases.size() - 1; i++)
    {
        file << "BIAS " << i + 1;
        for (auto bias : m_biases[i])
        {
            file << " " << bias;
        }
        file << std::endl;
    }

    file << std::endl;

    file << "#######################" << std::endl;
    file << "# criação de ligações #" << std::endl;
    file << "#######################" << std::endl;

    file << std::endl;

    file << "#" << std::endl;
    file << "# LIGACAO de_camada de_neuronio para_camada para_neuronio peso" << std::endl;
    file << "#" << std::endl;

    for (size_t i = 0; i < m_topologia.size() - 1; i++)
    {
        file << std::endl;
        file << "# Da camada " << i << " para a camada " << i + 1 << std::endl
             << std::endl;
        for (size_t j = 0; j < m_topologia[i]; j++)
            for (size_t k = 0; k < m_topologia[i + 1]; k++)
            {
                file << "LIGACAO " << i << " " << j << " " << i + 1 << " " << k << " " << m_pesos[i][j][k] << std::endl;
            }
    }

    file.close();

    return true;
} // salvar_rede

namespace
{
    struct set
    {
        int index;
        int qtd;
    };

    bool setcomp(set a, set b)
    {
        return a.index < b.index;
    }
}

bool Sequencial::carregar_rede(const std::string &caminho, func funcao_ativacao_oculta)
{
    this->funcao_ativacao_oculta = funcao_ativacao_oculta;

    std::ifstream file(caminho);

    if (!file.is_open())
        return false;

    std::string line;
    std::string keyword;

    // --- PASSADA 1: LER A ESTRUTURA E ALOCAR MEMÓRIA ---
    std::vector<set> temp_top;

    while (getline(file, line))
    {
        if (line.empty() || line[0] == '#')
            continue;

        std::stringstream ss(line);
        ss >> keyword;

        if (keyword == "CAMADA")
        {
            int index, neuronios;
            ss >> index >> neuronios;

            temp_top.push_back({index, neuronios});
        }
        else if (keyword == "TIPO_SAIDA")
        {
            std::string tipo;
            ss >> tipo;

            if (tipo == "SCE")
            {
                m_camada_saida = std::make_unique<SoftmaxCrossEntropy>();
            }
            else // valor padrão
            {
                m_camada_saida = std::make_unique<LinearMeanSquareError>();
            }
        }
    }

    if (temp_top.size() < 2)
        return false;

    sort(temp_top.begin(), temp_top.end(), setcomp);

    m_topologia.resize(temp_top.size());
    for (size_t i = 0; i < temp_top.size(); i++)
    {
        m_topologia[i] = temp_top[i].qtd;
    }

    // alocando memória nos vetores

    m_pesos.resize(m_topologia.size() - 1);
    m_biases.resize(m_topologia.size() - 1);
    m_gradientes_pesos.resize(m_topologia.size() - 1);
    m_gradientes_biases.resize(m_topologia.size() - 1);

    for (int i = 0; i < m_topologia.size() - 1; i++)
    {
        m_biases[i].resize(m_topologia[i + 1], 0.0);
        m_gradientes_biases[i].resize(m_topologia[i + 1], 0.0);

        m_pesos[i].resize(m_topologia[i]);
        m_gradientes_pesos[i].resize(m_topologia[i]);

        for (int j = 0; j < m_topologia[i]; j++)
        {
            m_pesos[i][j].resize(m_topologia[i + 1], 0.0);
            m_gradientes_pesos[i][j].resize(m_topologia[i + 1], 0.0);
        }
    }

    // --- PASSADA 2: FAZENDO AS LIGAÇÕES, ATRIBUINDO PESOS E BIASES ---

    file.clear();
    file.seekg(0, std::ios::beg);

    while (getline(file, line))
    {
        if (line.empty() || line[0] == '#')
            continue;

        std::stringstream ss(line);

        ss >> keyword;

        if (keyword == "BIAS")
        {
            int camada;
            ss >> camada;

            if (camada < 1 || camada >= m_topologia.size())
                return false;

            double bias;

            for (int i = 0; i < m_topologia[camada]; i++)
            {
                ss >> bias;
                m_biases[camada - 1][i] = bias;
            }
        }
        else if (keyword == "LIGACAO")
        {
            int de_camada, de_neuronio, para_camada, para_neuronio;
            double peso;

            ss >> de_camada >> de_neuronio >> para_camada >> para_neuronio >> peso;

            if (de_camada >= m_topologia.size() - 1 || de_camada + 1 != para_camada ||
                de_neuronio >= m_topologia[de_camada] || para_neuronio >= m_topologia[para_camada])
            {
                return false;
            }

            m_pesos[de_camada][de_neuronio][para_neuronio] = peso;
        }
    }

    file.close();

    return true;
} // carregar_rede

//
// OPERADORES
//

Sequencial &Sequencial::operator=(const Sequencial &other)
{
    if (this == &other)
        return *this;

    // Membros mais "simples"
    this->m_topologia = other.m_topologia;
    this->m_pesos = other.m_pesos;
    this->m_biases = other.m_biases;
    this->funcao_ativacao_oculta = other.funcao_ativacao_oculta;

    this->m_gradientes_pesos = other.m_gradientes_pesos;
    this->m_gradientes_biases = other.m_gradientes_biases;

    // Membros do otimizador Adam
    this->m_pesos_m = other.m_pesos_m;
    this->m_pesos_v = other.m_pesos_v;
    this->m_biases_m = other.m_biases_m;
    this->m_biases_v = other.m_biases_v;
    this->m_timestep = other.m_timestep;

    // Lidar com std::unique_ptr usando o método clone()
    if (other.m_camada_saida)
    {
        this->m_camada_saida = other.m_camada_saida->clone();
    }
    else
    {
        m_camada_saida.reset();
    }

    // As variáveis de chache são temporárias, portanto não precisam ser copiadas
    this->m_logits.clear();
    this->m_ativacoes.clear();

    return *this;
}