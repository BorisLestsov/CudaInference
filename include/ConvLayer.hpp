#ifndef CUDA_PROJ_CONVLAYER_CUH
#define CUDA_PROJ_CONVLAYER_CUH

#include "Layer.hpp"
#include "Tensor.hpp"

#include <string>
#include <memory>

#include <cublas.h>
#include <cublas_v2.h>


class ConvLayer: public Layer {
public:
    ConvLayer(cublasHandle_t& cublas_handle_p, const std::string& w_path, int stride=1, int pad=0, bool bias=true);
    ~ConvLayer();

    void forward();

    void set_input(std::shared_ptr<Tensor<float>> input);
    std::shared_ptr<Tensor<float>> get_output();

private:
    std::shared_ptr<Tensor<float>> _input, _w, _b, _res;
    std::shared_ptr<Tensor<float>> _imcol, _wcol, _bcol, _tmp;
    std::vector<float> data_b;
    int Hi;
    int Wi;
    int Ho;
    int Wo;
    int batch_size, N, C, H, W, _pad, _stride;
    int m;
    int n;
    int k;
    bool input_set, _bias;
    cublasHandle_t& cublas_handle;

};

#endif //CUDA_PROJ_CONVLAYER_CUH
