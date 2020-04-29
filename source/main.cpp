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
    } catch (const std::exception& e) {
        std::cout << "Exception: " << e.what() << std::endl;
    }

    
    try {
        InitializeCUDA(0);
        cublasHandle_t cublas_handle;
        checkCublasErrors(cublasCreate(&cublas_handle));

        PPMImage* img;
        img = readPPM(parser.retrieve<std::string>("input").c_str());
        // int n=1, c=3, h=224, w=224;
        // float* float_data_ptr = (float*) malloc(n*c*h*w*sizeof(float));
        // uchar* uchar_ptr = (uchar*) img->data;
        // for (int i = 0; i < h*w*c; ++i){
        //     float_data_ptr[i] = (float) (uchar_ptr[i]);
        // }

        int n=2, c=3, h=5, w=5;
        float* float_data_ptr = (float*) malloc(n*c*h*w*sizeof(float));
        for (int i = 0; i < n*h*w*c; ++i){
            float_data_ptr[i] = (float) (i+1);
        }

        // Net net(cublas_handle);
        // std::vector<Layer*> layers;
        // layers.push_back(new LinearLayer());
        // layers.push_back(new LinearLayer());
        // net.add_layer(layers[0]);
        // net.add_layer(layers[1]);

        Tensor<float>* input = new Tensor<float>({n, c, h, w});
        Tensor<float>* output;
        input->from_cpu(float_data_ptr);

        LinearLayer* linear = new LinearLayer(cublas_handle, "kek");

        input->reshape({input->size()[0], input->size()[1] * input->size()[2] * input->size()[3]});
        linear->set_input(input);
        linear->forward();
        output = linear->get_output();

        float* cpu_result = (float*) malloc(output->count()*sizeof(float));
        output->to_cpu(cpu_result);

        for (int i = 0; i < output->count(); ++i){
            std::cout << i << "  " << cpu_result[i] << std::endl;
        }


        // TODO: delete all!!

        free(img->data);
        free(img);
    } catch (const std::exception& e) {
        std::cout << "Exception: " << e.what() << std::endl;
    } catch (const std::string& e) {
        std::cout << "Exception: " << e << std::endl;
    }
}


