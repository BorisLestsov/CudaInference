#include <cublas.h>
#include <cublas_v2.h>

#include "Net.hpp"


Net::Net(cublasHandle_t& cublas_handle_p):
        cublas_handle(cublas_handle_p)
{}


void Net::add_layer(Layer* layer){
    layers.push_back(layer);
}

Tensor<float>* Net::forward(Tensor<float>* data){
    Tensor<float>* inter;
    for (int i = 0; i < layers.size(); ++i){
        layers[i]->forward();

    }
    return inter;
}
