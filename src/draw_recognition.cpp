#include <vector>
#include <string>
#include <algorithm>

#include <ftxui/component/component.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/dom/elements.hpp>
#include <ftxui/dom/canvas.hpp>
#include <ftxui/component/event.hpp>

#include "rede_neural.h"

const char* model_path = "data/models/numbr_rec_model.txt";
using matriz = std::vector<std::vector<u_char>>;
using namespace ftxui;

void insertCircle (matriz&, int, int, double);
void deleteCircle (matriz&, int, int, double);
nn::Vetor padronizar (const matriz&);

int main()
{
  // Tamanho da nossa area de desenho
  const int LARGURA = 140;
  const int ALTURA  = 140;

  double raio_pincel = 6;

  matriz tela_pixels(
    ALTURA,
    std::vector<u_char>(LARGURA, 0)
  );

  int mouse_x = 0;
  int mouse_y = 0;

  bool desenhando = false;
  bool apagando   = false;
  bool preveu = true;

  auto screen = ScreenInteractive::FitComponent();

  auto rede = nn::Sequencial(model_path);
  nn::Vetor previsoes(10, 0.0);
  nn::Vetor entrada(28*28, 0.0);

  // O Renderer é a funcao que desenha tudo na tela a cada frame
  auto renderer_pintura = Renderer([&]{
    auto c = Canvas(LARGURA, ALTURA);

    // pincel
    c.DrawPointCircle(mouse_x, mouse_y, raio_pincel);

    for (int y = 0; y < ALTURA; ++y)
    for (int x = 0; x < LARGURA; ++x)
    {
      if (tela_pixels[y][x] == 1)
      {
        c.DrawPoint(x, y, true);
      }
    }
    
    return window (text(" pAInt 2000 "), canvas(std::move(c)) | flex) ;
  });

  auto painel_direito = Renderer([&]{
    Elements barras_previsao;

    for (int i = 0; i < 10; ++i)
    {
      barras_previsao.push_back(
        hbox({
          text(std::to_string(i) + "|"),
          gauge(previsoes[i]) | color(Color::White),
          text(" " + std::to_string(int(previsoes[i] * 100)) + "%") | size(WIDTH, EQUAL, 5),
        })
      );
    }

    auto instrucoes = vbox({
      text("Botão Direito do Mouse: Apagar"),
      text("Botão Esquerdo do Mouse: Desenhar"),
      text("Roda do Mouse: Tamanho do Pincel"),
      text("pressione 'q' para sair | 'c' : Resetar Tela"),
    });

    return window(text(" previsão "), vbox({
      vbox(barras_previsao),
      filler(),
      paragraph("OBS: é mais fácil de reconhecer com um desenho maior"),
      filler(),
      separator(),
      instrucoes,
    }) | size(WIDTH, GREATER_THAN, 40));
  });

  auto layout_final = Container::Horizontal({
    renderer_pintura,
    painel_direito
  });
  
  auto listener = CatchEvent(layout_final, [&](Event e) {
    if (e.is_mouse()) 
    {
      mouse_x = (e.mouse().x - 1) * 2;
      mouse_y = (e.mouse().y - 1) * 4;

      if (e.mouse().button == Mouse::Left)
      {
        if (e.mouse().motion == Mouse::Pressed) desenhando = true;
        if (e.mouse().motion == Mouse::Released) desenhando = false;
      }

      if (e.mouse().button == Mouse::Right)
      {
        if (e.mouse().motion == Mouse::Pressed) apagando = true;
        if (e.mouse().motion == Mouse::Released) apagando = false;
      }

      if (e.mouse().button == Mouse::WheelUp && raio_pincel <= 20) raio_pincel += 0.5;
      if (e.mouse().button == Mouse::WheelDown && raio_pincel > 0) raio_pincel -= 0.5;

      if (desenhando)
      {
        preveu = false;
        insertCircle (tela_pixels, mouse_x, mouse_y, raio_pincel);
      }

      if (apagando)
      {
        preveu = false;
        deleteCircle (tela_pixels, mouse_x, mouse_y, raio_pincel);
      }
    }

    if (!desenhando && !apagando && !preveu)
    {
      entrada = padronizar(tela_pixels);
      previsoes = rede.feed_forward(entrada);
      preveu = true;
    }
    
    if (e.is_character())
    {
      if (e.character() == "q") 
      {
        screen.Exit();
        return true;
      }

      if (e.character() == "c")
      {
        for (size_t y = 0; y < tela_pixels.size(); ++y)
        for (size_t x = 0; x < tela_pixels[y].size(); ++x)
          tela_pixels[y][x] = 0;

        preveu = true;
        return true;
      }
    }

    return false;
  });

  screen.Loop(listener);

  return 0;
}

void insertCircle (matriz& tela, int a, int b, double raio)
{
  for (size_t y = 0; y < tela.size(); ++y)
  for (size_t x = 0; x < tela[y].size(); ++x)
  {
    // Verifica se o ponto (x, y) esta dentro do circulo
    if ((x - a) * (x - a) + (y - b) * (y - b) <= raio * raio)
    {
      tela[y][x] = 1;
    }
  }
}

void deleteCircle (matriz& tela, int a, int b, double raio)
{
  for (size_t y = 0; y < tela.size(); ++y)
  for (size_t x = 0; x < tela[y].size(); ++x)
  {
    // Verifica se o ponto (x, y) esta dentro do circulo
    if ((x - a) * (x - a) + (y - b) * (y - b) <= raio * raio)
    {
      tela[y][x] = 0;
    }
  }
}

nn::Vetor padronizar(const matriz& tela)
{
    const int TARGET_DIM = 28;
    nn::Vetor saida(TARGET_DIM * TARGET_DIM, 0.0); // Vetor já com o tamanho final

    const double source_height = tela.size();
    // Garante que não haverá erro se a tela estiver vazia
    const double source_width = tela.empty() ? 0 : tela[0].size();

    if (source_width == 0 || source_height == 0) {
        return saida; // Retorna o vetor zerado se a tela não tiver dimensões
    }
    
    // Calcula a proporção entre a imagem original e a imagem de destino
    const double ratio_y = source_height / TARGET_DIM;
    const double ratio_x = source_width / TARGET_DIM;

    // Itera sobre a grade de destino de 28x28
    for (int target_y = 0; target_y < TARGET_DIM; ++target_y) {
        for (int target_x = 0; target_x < TARGET_DIM; ++target_x) {
            
            // Calcula qual "bloco" na imagem original corresponde a este pixel de destino
            int source_y_start = static_cast<int>(target_y * ratio_y);
            int source_x_start = static_cast<int>(target_x * ratio_x);
            int source_y_end = static_cast<int>((target_y + 1) * ratio_y);
            int source_x_end = static_cast<int>((target_x + 1) * ratio_x);

            double soma = 0.0;
            int num_pixels_no_bloco = 0;

            // Itera sobre o bloco correspondente na imagem original
            for (int y = source_y_start; y < source_y_end; ++y) {
                for (int x = source_x_start; x < source_x_end; ++x) {
                    // Verificação de segurança para não acessar fora dos limites
                    if (y < source_height && x < source_width) {
                        soma += tela[y][x];
                        num_pixels_no_bloco++;
                    }
                }
            }
            
            // Calcula a média e a atribui à posição correta no vetor de saída
            if (num_pixels_no_bloco > 0) {
                saida[target_y * TARGET_DIM + target_x] = soma / num_pixels_no_bloco;
            }
        }
    }

    return saida;
}