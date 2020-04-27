#ifndef CUDA_PROJ_LINEARLAYER_CUH
#define CUDA_PROJ_LINEARLAYER_CUH

#include "Layer.hpp"

class LinearLayer: public Layer {
public:
    LinearLayer();

    void forward();

private:

};

#endif //CUDA_PROJ_LINEARLAYER_CUH
