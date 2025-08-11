#include "rede_neural.h"

#include <fstream>
#include <vector>
#include <iostream>
#include <random>

using namespace std;

const char* labels_file_path = "data/dataset/train-labels.idx1-ubyte";
const char* images_file_path = "data/dataset/train-images.idx3-ubyte";

const int numero_imagens  = 60000;
const int tamanho_imagem = 28 * 28;

int main()
{
    ifstream labels_file (labels_file_path, ios::binary);
    ifstream images_file (images_file_path, ios::binary);

    if (!labels_file.is_open() || !images_file.is_open())
    {
        cout << "Erro ao abrir os arquivos do dataset MNIST!" << endl;
        return 1;
    }

    vector<vector<double>> todas_imagens;
    vector<vector<double>> todos_rotulos;

    // pular os cabeçalhos
    images_file.seekg(16);
    labels_file.seekg(8);

    cout << "Lendo " << numero_imagens << " imagens..." << endl;

    for (int i = 0; i < numero_imagens; ++i)
    {
        // buffer para ler os bytes de uma imagem
        vector<u_char> buffer_imagem(tamanho_imagem);
        u_char buffer_rotulo;

        // lê os 784 bytes do arquivo e coloca no buffer
        images_file.read(reinterpret_cast<char*>(buffer_imagem.data()), tamanho_imagem);
        labels_file.read(reinterpret_cast<char*>(&buffer_rotulo), 1);

        vector<double> imagem;

        imagem.reserve(tamanho_imagem);

        for (auto pixel : buffer_imagem)
        {
            // normalizar para [0.0, 1.0]
            imagem.push_back(static_cast<double>(pixel) / 255.0);
        }

        vector<double> rotulo(10, 0.0);
        rotulo[buffer_rotulo] = 1.0;

        todas_imagens.push_back(imagem);
        todos_rotulos.push_back(rotulo);
    }

    images_file.close();
    labels_file.close();

    cout << "Leitura concluída!" << endl << endl;

    cout << "Embaralhando imagens..." << endl;

    vector<int> indices(numero_imagens);
    iota(indices.begin(), indices.end(), 0);

    random_device rd;
    mt19937 g(rd());
    shuffle(indices.begin(), indices.end(), g);

    vector<vector<double>> imagens_embaralhadas;
    vector<vector<double>> rotulos_embaralhados;
    imagens_embaralhadas.reserve(numero_imagens);
    rotulos_embaralhados.reserve(numero_imagens);
    
    for (int i : indices)
    {
        imagens_embaralhadas.push_back(todas_imagens[i]);
        rotulos_embaralhados.push_back(todos_rotulos[i]);
    }

    todas_imagens = imagens_embaralhadas;
    todos_rotulos = rotulos_embaralhados;

    imagens_embaralhadas.clear();
    rotulos_embaralhados.clear();

    cout << "Embaralhamento concluído!" << endl << endl;

    nn::Matriz entradas_treino, saidas_treino, entradas_validacao, saidas_validacao;

    // separando os dados em entradas de treino e entradas de validação
    for (int i = 0; i < numero_imagens * 0.2; ++i)
    {
        entradas_validacao.push_back(todas_imagens[i]);
        saidas_validacao.push_back(todos_rotulos[i]);
    }
    
    for (size_t i = entradas_validacao.size(); i < numero_imagens; ++i)
    {
        entradas_treino.push_back(todas_imagens[i]);
        saidas_treino.push_back(todos_rotulos[i]);
    }

    todas_imagens.clear();
    todos_rotulos.clear();

    nn::Sequencial numbr_rec({tamanho_imagem, 32, 32, 10}, "SCE");

    numbr_rec.train(entradas_treino, saidas_treino, entradas_validacao, saidas_validacao, 0.001, 10, 0.2, 1e-5);
    numbr_rec.salvar_rede("data/models/numbr_rec_model.txt");

    return 0;
}