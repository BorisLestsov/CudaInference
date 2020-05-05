#ifndef CUDA_PROJ_RELULAYER_CUH
#define CUDA_PROJ_RELULAYER_CUH

#include "Layer.hpp"
#include "Tensor.hpp"

#include <string>
#include <memory>


class ReluLayer: public Layer {
public:
    ReluLayer();
    ~ReluLayer();

    void forward();

    void set_input(std::shared_ptr<Tensor<float>> input);
    std::shared_ptr<Tensor<float>> get_output();

private:
    std::shared_ptr<Tensor<float>> _input, _res;
    int batch_size;

};

#endif //CUDA_PROJ_RELULAYER_CUH
