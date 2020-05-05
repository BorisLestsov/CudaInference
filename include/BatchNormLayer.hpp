#ifndef CUDA_PROJ_BATCHNORMLAYER_CUH
#define CUDA_PROJ_BATCHNORMLAYER_CUH

#include "Layer.hpp"
#include "Tensor.hpp"

#include <string>


class BatchNormLayer: public Layer {
public:
    BatchNormLayer(const std::string& path, int batch_size=1);
    ~BatchNormLayer();

    void forward();

    void set_input(Tensor<float>* input);
    Tensor<float>* get_output();

private:
    Tensor<float>* _input, *_res;
    Tensor<float>* _w, *_b, *_rm, *_rv;
    int batch_size, C;
    int Hi;
    int Wi;
    int Ho;
    int Wo;

    std::vector<float> data_w, data_b, data_rm, data_rv;

};

#endif //CUDA_PROJ_BATCHNORMLAYER_CUH
