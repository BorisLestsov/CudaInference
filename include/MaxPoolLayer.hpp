#ifndef CUDA_PROJ_MAXPOOLLAYER_CUH
#define CUDA_PROJ_MAXPOOLLAYER_CUH

#include "Layer.hpp"
#include "Tensor.hpp"

#include <string>
#include <memory>


class MaxPoolLayer: public Layer {
public:
    MaxPoolLayer(int ker_size=2, int stride=1, int pad=0);
    ~MaxPoolLayer();

    void forward();

    void set_input(std::shared_ptr<Tensor<float>> input);
    std::shared_ptr<Tensor<float>> get_output();

private:
    std::shared_ptr<Tensor<float>> _input, _res;
    int batch_size, _stride, _pad, H, W, C;
    int Hi;
    int Wi;
    int Ho;
    int Wo;

};

#endif //CUDA_PROJ_MAXPOOLLAYER_CUH
