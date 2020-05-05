#ifndef CUDA_PROJ_AVGPOOLLAYER_CUH
#define CUDA_PROJ_AVGPOOLLAYER_CUH

#include "Layer.hpp"
#include "Tensor.hpp"

#include <string>


class AvgPoolLayer: public Layer {
public:
    AvgPoolLayer(int batch_size=1, int ker_size=2, int stride=1, int pad=0);
    ~AvgPoolLayer();

    void forward();

    void set_input(Tensor<float>* input);
    Tensor<float>* get_output();

private:
    Tensor<float>* _input, *_res;
    int batch_size, _stride, _pad, H, W, C;
    int Hi;
    int Wi;
    int Ho;
    int Wo;

};

#endif //CUDA_PROJ_AVGPOOLLAYER_CUH
