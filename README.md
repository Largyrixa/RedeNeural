# Rede Neural
Este projeto implementa uma rede neural de uso genérico

## Como Utilizar

```C++
#include "rede_neural.h"
// resto do código
```
***Nota**: Certifique-se de que o arquivo `rede_neural.cpp` está incluso no comando de compilação*

## Exemplo de Implementação
O melhor exemplo que tenho é o meu repositório do [Snake Game](https://github.com/Largyrixa/SnakeGame.git)

## Arquivos de Descrição de Rede
Para facilitar a visualização de como a rede neural funciona e talvez até editar algumas ligações, criei uma sintaxe de arquivos que descreve uma rede neural da seguinte maneira:

| Palavra-Chave | Parâmetros | Descrição |
| :-----------: | :--------: | :--------: |
| `CAMADA_ENTRADA`| I N | Cria uma camada de N neurônios de entrada |
| `CAMADA_OCULTA` | I N | Cria uma camada oculta de índice I com N neurônios |
| `CAMADA_SAIDA` | I N | Cria uma camada de N neurônios de saída |
| `BIAS`          | I BIAS_0 BIAS_1 ... BIAS_N | Adiciona os bias da camada I |
| `LIGACAO` | I_in J_in I_out J_out peso | Adiciona uma ligação do neurônio J_in da camada I_in para o neurônio J_out da camada I_out |

***Nota:** Comentários podem ser feitos adicionando um `#` no início da linha*

Um exemplo de rede descrita com essa sintaxe pode ser visto em `data/exemplo_rede.txt` que forma a seguinte rede:

<img src="images/exemplo.png">