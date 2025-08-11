#ifndef _REDE_NEURAL_H
#define _REDE_NEURAL_H

#include "camadas_saida.h"
#include <vector>
#include <string>
#include <functional>
#include <memory>

namespace nn
{
  using Matriz = std::vector<std::vector<double>>;
  using Vetor = std::vector<double>;

  struct func
  {
    const char* nome;
    std::function<double(double)> funcao;
    std::function<double(double)> derivada;
  };

  extern std::unique_ptr<CamadaSaida> camada_saida_padrao;
  extern func ReLU;
  extern func tanh;
  extern func sigmoid;

  // Implementação de uma rede neural sequencial
  class Sequencial
  {
  public:
    /*
    Construtor: define a topologia da rede.
    A topologia é um vetor de inteiros que define quantos neurônios existem
    em uma camada, começando pela camada de entrada.


    @tparam topologia topologia da rede neural
    Exemplo: {3, 5, 6, 2} cria uma rede com:

    - 3 neurônios de entrada

    - 5 neurônios na primeira camada oculta

    - 6 neurônios na segunda camada oculta

    - 2 neurônios de saída


    @tparam funcao_ativacao_oculta do tipo func<double>
    @tparam camada_saida do tipo string

    Implementações de camadas de saída:
    "LMSE" - Linear Mean Square Error
    "SCE" - Softmax Cross Entropy
    */
    Sequencial(
        const std::vector<size_t> &topologia,
        std::string camada_saida_str,
        func funcao_ativacao_oculta
      );

    /*
    Construtor a partir de um arquivo de descrição de rede
    */
    Sequencial(const std::string &caminho);

    /*
    =====================================
      MÉTODOS DE FUNCIONALIDADE DA REDE
    =====================================
    */

    /*
    Calcula o resultado da rede para uma dada entrada.
    O vetor de entrada deve ter o mesmo tamanho da camada de entrada da rede.
    Retorna um vetor com os valores da camada de saída.
    */
    Vetor feed_forward(const Vetor &entradas) const;

    /*
    Função para treinar a rede neural com dados pré-estabelecidos
    @tparam entradas_treino todas as entradas a serem testadas
    @tparam saidas_treino saídas esperadas no treino
    @tparam entradas_validacao entradas que serão usadas para fazer a validação da rede, evitando overfiting
    @tparam saidas_validacao saídas que serão usadas para fazer a validação da rede, evitadno overfiting
    @tparam taxa_aprendizagem taxa de aprendizagem da rede: um número baixo pode levar a uma precisão maior, mas demorar mais,
    o inverso se aplica para números maiores
    @tparam janela_analise é a quantidade de losses que serão analisados para verificar se há uma estabilização, por padrão
    é 100
    @tparam target_loss loss alvo, por padrão é 0 o que leva a uma parada do treino somente quando o loss se estabiliza
    @tparam threshold devio padrão mínimo para acabar o treinamento (identifica se o loss estabilizou), por padrão é 1e-5
    */
    void train(const std::vector<Vetor> &entradas_treino, const std::vector<Vetor> &saidas_treino,
               const std::vector<Vetor> &entradas_validacao, const std::vector<Vetor> &saidas_validacao,
               double taxa_aprendizagem, size_t janela_analise = 100, double target_loss = 0.0, double threshold = 1e-5);

    /*
    Avalia o desempenho da rede
    */
    double calc_loss(const std::vector<Vetor> &entradas, const std::vector<Vetor> &saidas_esperadas) const;

    double calc_accuracy (const std::vector<Vetor>& entradas, const std::vector<Vetor>& saidas_esperadas) const;

    /*
    ===========================
      MÉTODOS DE PLASTICIDADE
    ===========================
    */

    /*
    Adiciona um neurônio a uma camada específica.
    @tparam index_camada deve ser maior que 0 (entrada) e menor que a última camada (saída).

    Por padrão, as novas conexões são inicializadas com pesos aleatórios pequenos.
    */
    void adicionar_neuronio(int index_camada);

    /*
    Remove um neurônio de uma camada específica.
    @tparam index_camada deve ser uma camada oculta.
    @tparam index_neuronio é a posição do neurônio a ser removido naquela camada.
    */
    void remover_neuronio(int index_camada, int index_neuronio);

    /*
    Muda as funções de ativação da rede
    */
    void set_func(func funcao_ativacao_oculta, std::unique_ptr<CamadaSaida> camada_saida);

    /*
    =====================================
      MÉTODOS PARA ALGORITMOS GENÉTICOS
    =====================================
    */

    // Define os pesos de uma camada específica
    void set_pesos(int index_camada, const Matriz &novos_pesos);

    // Define os biases de uma camada específica
    void set_biases(int index_camada, const Vetor &novos_biases);

    /*
    ===============================================
      MÉTODOS DE PERSISTENCIA (SALVAR E CARREGAR)
    ===============================================
    */

    bool salvar_rede(const std::string &caminho) const;
    bool carregar_rede(const std::string &caminho, func funcao_ativacao_oculta = ReLU);

    /*
    ===========
      GETTERS
    ===========
    */

    // Retorna uma matriz de pesos das ligações da camada index_camada para a index_camada + 1
    //
    // Exemplo: se M = matriz_pesos[index_camada], então M[i][j] é o peso da ligação
    // do neurônio i (da camada index_camada) para o neurônio j (da camada index_camada + 1)
    // Obs: a camada de saída não tem pesos (já que não existe próxima camada)
    const Matriz &get_pesos(int index_camada) const;

    // Retorna um vetor dos biases da camada index_camada
    // Obs: a camada 0 (entrada) não tem biases
    const Vetor &get_biases(int index_camada) const;

    // Retorna um vetor com a topologia da rede atual
    // Exemplo: uma rede com
    // - 2 neurônios na camada de entrada
    // - 3 na primeira camada oculta
    // - 5 na segunda camada oculta
    // - 2 na camada de saída
    // é representado por {2, 3, 5, 2}
    const std::vector<size_t> &get_topologia() const;

    Sequencial &operator=(const Sequencial &other);

  private:
    // A topologia define a estrutura da rede, ex: {3, 5, 2}
    std::vector<size_t> m_topologia;

    /*
    Os pesos são uma lista de matrizes
    m_pesos[0] é a matriz de pesos entre a camada de entrada (0) e a primeira camada oculta
    m_pesos[1] é a matriz de pesos entre a camada oculta 1 e 2.
    ...
    se W = m_pesos[i], então W[j][k] é o peso da conexão do neurônio j (da camada i)
    para o neurônio k (da camada i+1).
    */
    std::vector<Matriz> m_pesos;

    /*
    Os biases são uma lista de vetores.
    m_biases[0] é o vetor de biases para a primeira camada oculta (camada 1).
    m_biases[1] é para a segunda camada oculta.
    ...
    A camada de entrada não tem biases.
    */
    std::vector<Vetor> m_biases;

    // Membros para armazenar os gradientes gerados pelo backpropagate
    std::vector<Matriz> m_gradientes_pesos;
    std::vector<Vetor> m_gradientes_biases;

    // Membros para armazenar os valores intermediários do feed_forward
    mutable std::vector<Vetor> m_logits;    // armazena as somas ponderadas (antes da ativação)
    mutable std::vector<Vetor> m_ativacoes; // armazena as saídas ativadas de cada camada

    void backpropagate(const Vetor &saida_esperada);

    // otimizador Adam
    void otimizar(double taxa_aprendizagem, double beta1, double beta2, double epsilon);

    // Membros do Adam
    std::vector<Matriz> m_pesos_m, m_pesos_v;
    std::vector<Vetor> m_biases_m, m_biases_v;
    long m_timestep;

    func funcao_ativacao_oculta;
    std::unique_ptr<CamadaSaida> m_camada_saida;

    void inicializar_pesos();
    void inicializar_biases();
  };

} // namespace nn

#endif