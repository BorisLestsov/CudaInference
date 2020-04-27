#include <iostream>
#include <string>
#include <exception>
#include "argparse.hpp"
#include "common.h"
#include "helper_functions.cuh"

#include <cublas.h>
#include <cublas_v2.h>

#include "Net.hpp"
#include "Layer.hpp"
#include "LinearLayer.hpp"
#include "Tensor.hpp"


int main(int argc, const char** argv)
{

    argparse::ArgumentParser parser;
    try {
        parser.addArgument("--input", 1, false);
        parser.parse(argc, argv);
    } catch (std::exception e) {
        std::cout << "Exception: " << e.what() << std::endl;
    }

    
    try {
        PPMImage* img;
        img = readPPM(parser.retrieve<std::string>("input").c_str());

        cublasHandle_t cublas_handle;
        checkCublasErrors(cublasCreate(&cublas_handle));

        Net net(cublas_handle);
        std::vector<Layer*> layers;
        layers.push_back(new LinearLayer());
        layers.push_back(new LinearLayer());

        net.add_layer(layers[0]);
        net.add_layer(layers[1]);

        Tensor input, output;

        output = net.forward(input);




        free(img->data);
        free(img);
    } catch (std::exception e) {
        std::cout << "Exception: " << e.what() << std::endl;
    } catch (std::string e) {
        std::cout << "Exception: " << e << std::endl;
    }
}


