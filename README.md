# Bibioteca de Rede Neural em C++
Uma biblioteca C++ leve e flexível para construir, treinar e implantar redes neurais. A implementação e feita do zero, sem dependências de grandes frameworks, focando em facilidade de uso.

## Funcionalidades
- **Modelo Sequencial**: Crie redes neurais camada por camada.
- **Camadas de Saída Especializadas**:
    - `LMSE` (Linear Mean Square Error): Ideal para tarefas de **regressão**.
    - `SCE` (Softmax Cross-Entropy): Perfeita para tarefas de **classificação**.
- **Funções de Ativação Customizáveis**: Utilize a `ReLU` (padrão) para camadas ocultas ou defina suas próprias funções de ativação e suas derivadas.
- **Otimizador Adam**: Treinamento eficiente e moderno com o otimizador Adam, que ajusta a taxa de aprendizado de forma adaptativa.
- **Treinamento com Validação**: Monitore o `loss` em um conjunto de validação para evitar *overfitting* e salvar o melhor modelo.
- **Persistência de Modelo**: Salve os modelos treinados em arquivos de texto legíveis e carregue-os posteriormente para fazer previsões.

## Começando
O exemplo abaixo configura uma rede para o problema da porta lógica XOR.

``` C++
#include <iostream>
#include "rede_neural.h"

int main()
{
    // 1. Defina a topologia da rede
    // - 2 neurônios na camada de entrada
    // - 2 neurônios na camada oculta
    // - 2 neurônios na camada de saída
    // Softmax na camada de saída
    nn::Sequencial rede({2, 2, 2}, "SCE");

    // 2. Prepare os dados de treinamento
    std::vector<nn::Vetor> entradas_treino = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    std::vector<nn::Vetor> saidas_treino   = {{1, 0}, {0, 1}, {0, 1}, {1, 0}};

    // 3. Treine a rede
    // A rede usa o conjunto de treino para aprender e para validar
    std::cout << "Iniciando o treinamento..." << std::endl;
    rede.train(
        entradas_treino, saidas_treino, // Dados de treino
        entradas_treino, saidas_treino, // Dados de validação
        0.01,                           // Taxa de aprendizagem
        100,                            // Janela de análise para parada
        0.001,                          // Loss alvo
        1e-6                            // Threshold para estabilização
    );
    std::cout << "Treinamento concluído!" << std::endl;

    // 4. Teste a rede treinada
    std::cout << "\nResultados da Previsão:" << std::endl;
    for (const auto& entrada : entradas_treino) 
    {
        nn::Vetor previsao = rede.feed_forward(entrada);
        std::cout << "Entrada: [" << entrada[0] << ", " << entrada[1]
                  << "], Previsão: " << previsao[0] << std::endl;
    }

    // 5. Salve o modelo para uso futuro
    if (rede.salvar_rede("modelo_xor.txt"))
    {
        std::cout << "\nRede salva com sucesso em 'modelo_xor.txt'!" << std::endl;
    }

    return 0;
}
```

## Referência da API
`nn::Sequencial`

Esta é a classe principal que encapsula toda a funcionalidade da rede.

### Construtores
- `Sequencial(const std::vector<size_t>& topologia, std::string camada_saida_str, func funcao_ativacao_oculta = nn::ativ_oculta_padrao)`: Cria uma rede a partir de uma topologia.
    - `topologia`: Um vetor que define o número de neurônios em uma camada (entrada, ocultas..., saída).
    - `camada_saida_str`: O tipo de saída. `"LMSE"` e `"SCE"` estão implementados.
- Sequencial(const std::string& caminho, ...): Carrega uma rede a partir de um arquivo salvo.

### Métodos Principais
- `train(...)`: Inicia o processo de treinamento, Recebe dados de treino, dados de validação e hiperparâmetros, como taxa de aprendizado.
- `feed_forward(const Vetor& entradas)`: Retorna a previsão da rede. Os logits de cada camada ficam armazenados.
- `calc_los(...)`: Calcula o loss da rede para um conjunto de dados.
- `calc_accuracy(...)`: Calcula a precisão para tarefas de classificação.
- `salvar_rede(...)`: Salva a topologia, pesos e biases da rede em um arquivo.
- `carregar_rede(...)`: Carrega uma rede previamente salva.

## Exemplo: Reconhecimento de Dígitos no terminal
Este projeto inclui um exemplo bem legar de um reconhecedor de dígitos desenhados diretamente no terminal.

### Como Compilar e Executar
Este projeto utiliza o CMake para compilação, portanto, para sistemas Linux:
```bash
mkdir build
cd build
cmake ..
make

cd ..
./build/pAInt

``` 
A rede neural (carregada de `data/models/numbr_rec_model.txt`) analisará seu desenho em tempo real e exibirá as probabilidades de ser cada um dos dígitos (0 a 9).

<img src="images/exemplo_pAInt.png"/>

*Exemplo de uso do pAInt*

*Nota: por ser treinada com o MNIST, pode ser que a rede tenha dificuldades em reconhecer o seu desenho, tente fazer traços mais grossos e mais embaixo na tela*

Caso não queira gastar seus dons artisticos, há um programa em `build/exemplo` que pega 10.000 imagens do MNIST (não usadas no treinamento) para fazer previsões.

## Arquivos de Descrição de Rede Neural
Para o projeto, criei uma sintaxe para descrever redes neurais, ainda é um pouco primitivo.

| Comando  | Parâmetros | Descrição |
| :------: | :--------: | :-------: |
| `CAMADA` | i n        | Cria uma camada de index `i` com `n` neurônios (0 para entrada)|
| `TIPO_SAIDA` | TipoDaCamadaDeSaida | Define o tipo da camada de saída, como `SCE` (Softmax Cross-Entropy) ou `LMSE` (Linear Mean Square Error) |
| `BIAS`   | i BIAS_0 BIAS_1 ... BIAS_N | Define os biases da camada `i` |
| `LIGACAO` | de_camada de_neuronio para_camada para_neuronio peso | Cria e define o peso para a ligação de um neurônio com outro |

*Comentários podem ser feitos adicionando `#` ao início da linha*

A imagem abaixo representa uma rede que aproxima a porta lógica XOR, descrita em `data/models/xor_model.txt`.

<img src="images/exemplo.png"/>

*Exemplo de rede descrita com essa sintaxe*