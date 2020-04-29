#ifndef CUDA_PROJ_RELULAYER_CUH
#define CUDA_PROJ_RELULAYER_CUH

#include "Layer.hpp"
#include "Tensor.hpp"

#include <string>


class ReluLayer: public Layer {
public:
    ReluLayer(int el_size, int batch_size = 1);
    ~ReluLayer();

    void forward();

    void set_input(Tensor<float>* input);
    Tensor<float>* get_output();

private:
    Tensor<float>* _input, *_res;
    int batch_size, el_size;

};

#endif //CUDA_PROJ_RELULAYER_CUH
