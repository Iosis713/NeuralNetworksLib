cmake_minimum_required(VERSION 3.28)
project(NeuralNetwork)

set(FLAGS -Werror -Wall -Wpedantic -Wextra -Og)
set(CMAKE_CXX_COMPILER clang++-18)
set(CMAKE_CXX_EXTENSIONS OFF)

set(GMOCK_LIBRARIES /usr/lib/libgmock.a /usr/lib/libgmock_main.a)
find_package(GTest REQUIRED)

set(SRC     Source/InputNeuron.cpp
            Source/Neuron.cpp
            Source/Layer.cpp
            Source/NeuralNet.cpp
            )

add_executable(${PROJECT_NAME} main.cpp)
add_library(${PROJECT_NAME}-lib STATIC ${SRC})

target_compile_options(${PROJECT_NAME}  PRIVATE ${FLAGS})
target_compile_options(${PROJECT_NAME}-lib  PRIVATE ${FLAGS})

set_target_properties(${PROJECT_NAME}   PROPERTIES  CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
set_target_properties(${PROJECT_NAME}-lib   PROPERTIES  CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
