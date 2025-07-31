#include "rede_neural.h"
#include <random>
#include <cmath>
#include <fstream>
#include <string>
#include <sstream>

using namespace std;

// Construtor
RedeNeural::RedeNeural
(int inputs, int hiddens, int outputs):
    inputs(inputs),
    hiddens(hiddens),
    outputs(outputs) {
    pesos_ih.resize(hiddens, vector<double>(inputs));
    bias_h.resize(hiddens);
    pesos_ho.resize(outputs, vector<double>(hiddens));
    bias_o.resize(outputs);

    random_device rd;
    default_random_engine generator(rd());
    uniform_real_distribution<double> distribution(-1.0,1.0);
    
    //
    // gerar pesos e bias aleatórios (entre -1.0 e 1.0)
    //
    
    for (int i = 0; i < hiddens; i++) {
        for (int j = 0; j < inputs; j++) {
            pesos_ih[i][j] = distribution(generator);
        }
        bias_h[i] = distribution(generator);
    }

    for (int i = 0; i < outputs; i++) {
        for (int j = 0; j < hiddens; j++) {
            pesos_ho[i][j] = distribution(generator);
        }
        bias_o[i] = distribution(generator);
    }
}

// Função de processamento
vector<double> RedeNeural::feedFoward
(const vector<double>& entradas) const {
    // --- Etapa 1: Entrada -> Oculta ---

    /*
    Explicação do que está acontecendo:
    Estamos pegando a matriz entradas, que tem o tamanho da quantidade
    de neurônios e multiplicando pela matriz pesos_ih, que representa
    os pesos da conversão de neurônios de input para neuronios ocultos
    */

    vector<double> saidas_oculta(pesos_ih.size());

    for (size_t i = 0; i < pesos_ih.size(); ++i) {
        double soma_ponderada = 0.0;
        
        for (size_t j = 0; j < entradas.size(); ++j) {
            soma_ponderada += entradas[j] * pesos_ih[i][j];
        }

        soma_ponderada += bias_h[i];
        saidas_oculta[i] = tanh(soma_ponderada);
    }

    // --- Etapa 2: Oculta -> Saída ---

    /*
    Aqui estamos convertendo neurônios ocultos para a saída
    */

    vector<double> saidas_finais(pesos_ho.size());

    for (size_t i = 0; i < pesos_ho.size(); ++i) {
        double soma_ponderada = 0.0;

        for (size_t j = 0; j < saidas_oculta.size(); ++j) {
            soma_ponderada += saidas_oculta[j] * pesos_ho[i][j];
        }

        soma_ponderada += bias_o[i];
        saidas_finais[i] = tanh(soma_ponderada);
    }

    return saidas_finais;
}

RedeNeural& RedeNeural::operator=(const RedeNeural& other) {
    if (this != &other) {
        this->bias_h = other.bias_h;
        this->bias_o = other.bias_o;
        this->hiddens = other.hiddens;
        this->inputs = other.inputs;
        this->outputs = other.outputs;
        this->pesos_ho = other.pesos_ho;
        this->pesos_ih = other.pesos_ih;
    }
    return *this;
}

bool RedeNeural::carregar_rede(const std::string& path) {
    ifstream file;
    file.open(path);

    if (!file.is_open()) return false;

    string line;
    string keyword;

    // --- PASSADA 1: LER A ESTRUTURA E ALOCAR MEMÓRIA ---
    int temp_inputs = 0, temp_hiddens = 0, temp_outputs = 0;

    while (getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;

        stringstream ss(line);
        ss >> keyword;

        if (keyword == "CAMADA_ENTRADA") {
            int index;
            ss >> index >> temp_inputs;
        } else if (keyword == "CAMADA_OCULTA") {
            int index;
            ss >> index >> temp_hiddens;
        } else if (keyword == "CAMADA_SAIDA") {
            int index;
            ss >> index >> temp_outputs;
        }
    }

    // Validação e alocação de memória
    if (temp_inputs == 0 || temp_hiddens == 0 || temp_outputs == 0) {
        file.close();
        return false;
    }

    this->inputs = temp_inputs;
    this->hiddens = temp_hiddens;
    this->outputs = temp_outputs;

    pesos_ih.resize(hiddens, vector<double>(inputs, 0.0));
    bias_h.resize(hiddens, 0.0);
    pesos_ho.resize(outputs, vector<double>(hiddens, 0.0));
    bias_o.resize(outputs, 0.0);

    // --- PASSADA 2: LER OS DADOS ---
    file.clear();
    file.seekg(0, ios::beg);

    while (getline(file, line)) {
        if (line.empty()) {
            continue;
        }

        stringstream ss(line);

        ss >> keyword;
        if (keyword == "BIAS") {
            int camada_idx;
            ss >> camada_idx;

            if (camada_idx == 1) {
                for (int i = 0; i < hiddens; i++) {
                    ss >> bias_h[i];
                }
            } else if (camada_idx == 2) {
                for (int i = 0; i < outputs; i++) {
                    ss >> bias_o[i];
                }
            }
        } else if (keyword == "LIGACAO") {
            int de_camada;
            int de_neuronio;
            int para_camada;
            int para_neuronio;
            double peso;

            ss >> de_camada >> de_neuronio >> para_camada >> para_neuronio >> peso;

            if (de_camada == 0 && para_camada == 1) {
                if (para_neuronio < hiddens && de_neuronio < inputs) {
                    pesos_ih[para_neuronio][de_neuronio] = peso;
                } else {
                    file.close();
                    return false;
                }
            } else if (de_camada == 1 && para_camada == 2) {
                if (para_neuronio < outputs && de_neuronio < hiddens) {
                    pesos_ho[para_neuronio][de_neuronio] = peso;
                } else {
                    file.close();
                    return false;
                }
            } else {
                file.close();
                return false;
            }
        }

    }


    return true;
}

bool RedeNeural::salvar_rede(const std::string& path) const{
    fstream file;
    file.open(path, fstream::out | fstream::trunc);

    if (!file.is_open()) return false;

    file << "#######################################" << endl;
    file << "# ARQUIVO DE DESCRIÇÃO DE REDE NEURAL #" << endl;
    file << "# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- #" << endl;
    file << "#         >>> REDE IAGO <<<           #" << endl;
    file << "#######################################" << endl;

    file << endl;

    file << "######################" << endl;
    file << "# criação de camadas #" << endl;
    file << "######################" << endl;

    file << endl;

    file << "#" << endl; 
    file << "# camada index número_de_entradas" << endl;
    file << "#" << endl;

    file << endl;
    
    
    file << "CAMADA_ENTRADA" << " 0 " << inputs << endl;


    file << "CAMADA_OCULTA" << " 1 " << hiddens << endl;


    file << "CAMADA_SAIDA" << " 2 " << outputs << endl;
    file << endl;

    file << "#######################" << endl;
    file << "# definição de biases #" << endl;
    file << "#######################" << endl;
    file << endl;

    file << "#" << endl;
    file << "# BIAS camada bias_0 bias_1 ... bias_n-1" << endl;
    file << "#" << endl;
    file << endl;

    file << "BIAS 1";

    for (auto bias : bias_h) file << " " << bias;
    file << endl;
    
    file << "BIAS 2";
    
    for (auto bias : bias_o) file << " " << bias;
    file << endl << endl;

    file << "#######################" << endl;
    file << "# criação de ligações #" << endl;
    file << "#######################" << endl;

    file << endl;

    file << "#" << endl;
    file << "# LIGACAO de_camada de_neuronio para_camada para_neuronio peso" << endl;
    file << "#" << endl;

    file << endl;

    file << "# Ligações da Entrada (0) para a Oculta (1)" << endl;
    file << endl;

    for (int i = 0; i < inputs; ++i) 
    for (int j = 0; j < hiddens; ++j) {
        file << "LIGACAO 0 " << i << " 1 " << j << " " << pesos_ih[j][i] << endl;
    }
    file << endl;

    file << "# Ligações da Oculta (1) para a Saída (2)" << endl;
    file << endl;

    for (int i = 0; i < hiddens; ++i)
    for (int j = 0; j < outputs; ++j) {
        file << "LIGACAO 1 " << i << " 2 " << j << " " << pesos_ho[j][i] << endl;
    }

    file.close();
    
    return true;
}
