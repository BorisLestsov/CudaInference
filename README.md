# CudaInference
Cuda NN inference. Example: ResNet18 in source/main.cpp.


## Functionality implemented:
* Convolution (via im2col) - with/without bias, arbitrary padding, arbitrary stride. Uses cuBLAS and thrust
* Linear - with/without bias. Uses cuBLAS and thrust.
* BatchNorm.
* ReLU.
* MaxPool - arbitrary padding, arbitrary stride.
* AvgPool - arbitrary padding, arbitrary stride.
* Tensor operations:
    * common operations (+, -, *, \/).
    * transpose - arbitrary number of dimentions, arbitrary axes permutation.
    * reshape.


## Features:
* Inference works with arbitrary batch size.
* NN weights are read from files on the disk. `python` directory contains weights and scripts to save pretrained weights to the disk.
* Any ResNet can be implemented with this functionality.
* Result is fully equivalent to Pytorch forward pass.
* Input image must:
    * be RGB image with 3 channels
    * be in PPM format
    * be exactly 224x224


## Build:
```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release .. 
make -j
./Release/cuda_proj --input ../images/cat.ppm --weights_dir ../python/weights/ --batch_size 16 --iters 100
```


## Usage:
```
./Release/cuda_proj --input ../images/cat.ppm --weights_dir ../python/weights/ --batch_size 16 --iters 100
```

The program will fill all inputs in the batch with image `../images/cat.ppm` and will perform `100` forward passes. Predicted labels and FPS will be prited.

## Benchmarks:
Benchmarks were done with batch_size == 16.

### FPS:
| Mode                                                                  | FPS  |
|-----------------------------------------------------------------------|------|
| CPU (Intel(R) Xeon(R) CPU E5-2620 v4 @ 2.10GHz) (Pytorch, 4 threads)  | 48   |
| CPU (Intel(R) Xeon(R) CPU E5-2620 v4 @ 2.10GHz) (Pytorch, 16 threads) | 81   |
| GPU (GeForce GTX 1080 Ti) (Pytorch)                                   | 2050 |
| GPU (GeForce GTX 1080 Ti) (This repo)                                 | 445  |

### Memory:
| Mode      | Memory usage |
|-----------|--------------|
| Pytorch   | 1317 MB      |
| This Repo | 2571 MB      |
