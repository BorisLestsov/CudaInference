#ifndef CUDA_PROJ_BATCHNORMLAYER_CUH
#define CUDA_PROJ_BATCHNORMLAYER_CUH

#include "Layer.hpp"
#include "Tensor.hpp"

#include <string>
#include <memory>


class BatchNormLayer: public Layer {
public:
    BatchNormLayer(const std::string& path);
    ~BatchNormLayer();

    void forward();

    void set_input(std::shared_ptr<Tensor<float>> input);
    std::shared_ptr<Tensor<float>> get_output();

private:
    std::shared_ptr<Tensor<float>> _input, _res;
    std::shared_ptr<Tensor<float>> _w, _b, _rm, _rv;
    int batch_size, C;
    int Hi;
    int Wi;
    int Ho;
    int Wo;

    std::vector<float> data_w, data_b, data_rm, data_rv;

};

#endif //CUDA_PROJ_BATCHNORMLAYER_CUH
