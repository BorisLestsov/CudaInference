#ifndef CUDA_PROJ_LINEARLAYER_CUH
#define CUDA_PROJ_LINEARLAYER_CUH

#include "Layer.hpp"
#include "Tensor.hpp"

#include <string>
#include <memory>

#include <cublas.h>
#include <cublas_v2.h>


class LinearLayer: public Layer {
public:
    LinearLayer(cublasHandle_t& cublas_handle_p, const std::string& w_path, bool bias=true);
    ~LinearLayer();

    void forward();

    void set_input(std::shared_ptr<Tensor<float>> input);
    std::shared_ptr<Tensor<float>> get_output();
    int get_output_dim();

private:
    int batch_size, input_dim, output_dim;
    std::shared_ptr<Tensor<float>> _input, _w, _b, _res, _tmp;
    cublasHandle_t& cublas_handle;
    std::vector<float> data_b;
    bool _bias;

};

#endif //CUDA_PROJ_LINEARLAYER_CUH
