#ifndef CUDA_PROJ_TENSOR_CUH
#define CUDA_PROJ_TENSOR_CUH

#include <vector>
#include <iostream>

typedef std::vector<int> Size;

template<typename T>
class Tensor {

public:
    Tensor(Size size_p);
    ~Tensor();

    int count();
    int ndim();
    Size size();
    void from_cpu(T* ptr);
    T* to_cpu();

    T* _ptr;
private:

    int _count;
    int _ndim;
    Size _size;

};


// IMPL

template<typename T>
Tensor<T>::Tensor(Size size_p):
    _size(size_p) 
{
    _count = 1;
    for (int i = 0; i < _size.size(); ++i) {
        _count *= _size[i];
    }
    cudaMalloc(&_ptr, _count*sizeof(T));
}

template<typename T>
Tensor<T>::~Tensor()
{
    cudaFree(_ptr);
}

template<typename T>
int Tensor<T>::ndim()
{
    return _ndim; 
}

template<typename T>
int Tensor<T>::count()
{
    return _count;
}

template<typename T>
void Tensor<T>::from_cpu(T* ptr)
{
    cudaMemcpy(_ptr, ptr, _count*sizeof(T), cudaMemcpyHostToDevice);
}

#endif //CUDA_PROJ_TENSOR_CUH
