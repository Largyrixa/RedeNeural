#include "rede_neural.h"

#include <fstream>
#include <iostream>
#include <vector>
#include <random>
#include <numeric>
#include <algorithm>

using namespace std;

const char* images_path = "data/dataset/t10k-images.idx3-ubyte";
const char* labels_path = "data/dataset/t10k-labels.idx1-ubyte";

const char* model_path  = "data/models/numbr_rec_model.txt";

int inverter_endian (int i)
{
    u_char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

int numero_imagens = 0;

void print_img (vector<double>);

int main()
{
    const char* images_path = "data/dataset/t10k-images.idx3-ubyte";
    const char* labels_path = "data/dataset/t10k-labels.idx1-ubyte";
    
    ifstream images_file (images_path, ios::binary);
    ifstream labels_file (labels_path, ios::binary);

    if (!images_file.is_open() || !labels_file.is_open())
    {
        cerr << "ERRO: FaLha ao abrir os arquivos." << endl;
        return 1;
    }
    
    // --- LEITURA DO CABEÇALHO ---
    int magic_number_images = 0;
    int n_rows = 0;
    int n_cols = 0;
    
    images_file.read(reinterpret_cast<char*>(&magic_number_images), sizeof(magic_number_images));
    images_file.read(reinterpret_cast<char*>(&numero_imagens), sizeof(numero_imagens));
    images_file.read(reinterpret_cast<char*>(&n_rows), sizeof(n_rows));
    images_file.read(reinterpret_cast<char*>(&n_cols), sizeof(n_cols));

    // Corrige o endianess
    magic_number_images = inverter_endian(magic_number_images);
    numero_imagens = inverter_endian(numero_imagens);
    n_rows = inverter_endian(n_rows);
    n_cols = inverter_endian(n_cols);

    cout << "--- CABEÇALHO DAS IMAGENS ---" << endl;
    cout << "MAGIC NUMBER: " << magic_number_images << endl;
    cout << "NÚMERO DE IMAGENS: " << numero_imagens << endl;
    cout << "LINHAS: " << n_rows << " COLUNAS: " << n_cols << endl;

    int magic_number_labels = 0;
    int numero_rotulos = 0;
    labels_file.read(reinterpret_cast<char*>(&magic_number_labels), sizeof(magic_number_labels));
    labels_file.read(reinterpret_cast<char*>(&numero_rotulos), sizeof(numero_rotulos));
    magic_number_labels = inverter_endian(magic_number_labels);
    numero_rotulos = inverter_endian(numero_rotulos);
    
    cout << "\n--- Cabeçalho dos Rótulos ---" << endl;
    cout << "Magic Number: " << magic_number_labels << endl;
    cout << "Número de Rótulos: " << numero_rotulos << endl;

    // --- LEITURA DOS DADOS ---
    int tamanho_imagem = n_rows * n_cols;
    vector<vector<double>> todas_imagens;
    vector<u_char>            todos_rotulos;
    
    cout << "\nLENDO " << numero_imagens << " IMAGENS..." << endl;

    for (int i = 0; i < numero_imagens; i++)
    {
        vector<u_char> buffer_imagem(tamanho_imagem);
        u_char buffer_rotulo;

        images_file.read(reinterpret_cast<char*>(buffer_imagem.data()), tamanho_imagem);
        labels_file.read(reinterpret_cast<char*>(&buffer_rotulo), 1);

        if (!images_file || !labels_file)
        {
            cerr << "ERRO: falha ao ler os dados da imagem/rótulo na iteração " << i << endl;
            return 1;
        }

        vector<double> imagem;
        imagem.reserve(tamanho_imagem);
        for (auto pixel : buffer_imagem)
        {
            imagem.push_back(static_cast<double>(pixel) / 255.0);
        }

        todas_imagens.push_back(imagem);
        todos_rotulos.push_back(buffer_rotulo);
    }
    cout << "LEITURA CONCLUÍDA COM SUCESSO! " << todas_imagens.size() << " IMAGENS CARREGADAS!" << endl;


    images_file.close();
    labels_file.close();

    vector<int> indices(numero_imagens);
    iota(indices.begin(), indices.end(), 0);

    random_device rd;
    mt19937 g(rd());
    shuffle(indices.begin(), indices.end(), g);

    nn::Sequencial rede(model_path);

    for (auto i : indices)
    {
        auto imagem_atual = todas_imagens[i];
        print_img (imagem_atual);
        cout << "'" << (int)todos_rotulos[i] << "'" << endl;
        auto previsao = rede.feed_forward(todas_imagens[i]);

        u_char r = 0;

        for (u_char i = 1; i < 10; i++)
        {
            if (previsao[i] > previsao[r]) r = i;
        }

        for (int i = 0; i < 10; i++)
        {
            cout << i << ": ";
            for (u_char j = 1; j < previsao[i]*50; j++) cout << "█";
            if (i == r) cout << "<- PREVISÃO";
            cout << endl;
        }
        cout << "pressione enter para continuar..." << endl;
        cin.get();
    }
}

void print_img (vector<double> img)
{
    const char full[] = "██";
    const char med1[] = "▓▓";
    const char med2[] = "▒▒";
    const char med3[] = "░░";
    const char dark[] = "  ";

    cout << "+--------------------------------------------------------+" << endl;
    for (u_char i = 0; i < 28; i++)
    {
        cout << "|";
        for (u_char j = 0; j < 28; j++)
        {
            auto pixel = img[i * 28 + j];

            if      (pixel < 0.20) cout << dark;
            else if (pixel < 0.40) cout << med3;
            else if (pixel < 0.60) cout << med2;
            else if (pixel < 0.80) cout << med1;
            else                   cout << full; 
        }
        cout << "|" << endl;
    }
    cout << "+--------------------------------------------------------+" << endl;
}