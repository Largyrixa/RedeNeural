cmake_minimum_required(VERSION 3.14)

project(rede_neural)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
  ftxui
  GIT_REPOSITORY https://github.com/ArthurSonzogni/FTXUI.git
  GIT_TAG        main
)
FetchContent_MakeAvailable(ftxui)

add_library(nn_sequencial src/nn_sequencial.cpp)
target_include_directories(nn_sequencial PUBLIC includes)

add_executable(pAInt src/pAInt.cpp)
target_link_libraries(pAInt PUBLIC nn_sequencial ftxui::component)

add_executable(exemplo src/exemplo.cpp)
target_link_libraries(exemplo PUBLIC nn_sequencial)