#ifndef CUDA_PROJ_TENSOR_CUH
#define CUDA_PROJ_TENSOR_CUH

#include <stdexcept>
#include <vector>
#include <iostream>
#include <compute_util.cuh>

typedef std::vector<int> Size;

template<typename T>
class Tensor {

public:
    Tensor(Size size_p);
    ~Tensor();

    int count() const;
    int ndim() const;
    const Size& size();
    void from_cpu(T* ptr);
    void to_cpu(T* ptr);
    Tensor& reshape(const Size& newsize);

    static Tensor* add_inplace(Tensor* src1, const Tensor* src2);

    T* _ptr;
private:

    int _count;
    int _ndim;
    Size _size;

};


// IMPL

template<typename T>
Tensor<T>::Tensor(Size size_p):
    _size(size_p),
    _ndim(size_p.size())
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
int Tensor<T>::ndim() const
{
    return _ndim; 
}

template<typename T>
int Tensor<T>::count() const
{
    return _count;
}

template<typename T>
void Tensor<T>::from_cpu(T* ptr)
{
    cudaMemcpy(_ptr, ptr, _count*sizeof(T), cudaMemcpyHostToDevice);
}

template<typename T>
void Tensor<T>::to_cpu(T* ptr)
{
    cudaMemcpy(ptr, _ptr, _count*sizeof(T), cudaMemcpyDeviceToHost);
}

template<typename T>
const Size& Tensor<T>::size()
{
    return _size;
}

template<typename T>
Tensor<T>& Tensor<T>::reshape(const Size& newsize)
{
    int newcount = 1;
    for (int i = 0; i < newsize.size(); ++i) {
        newcount *= newsize[i];
    }
    if (newcount != _count) {
        throw std::runtime_error("reshape wrong size");
    }
    _size = newsize;
    _count = newcount;
    _ndim = _size.size();
    return *this;
}


template<typename T>
Tensor<T>* Tensor<T>::add_inplace(Tensor* src1, const Tensor* src2)
{
    if (src1->count() != src2->count()){
        throw std::runtime_error("different size in Tensor::add_inplace");
    }
    cuda_add_inplace(src1->_ptr, src2->_ptr, src1->_ptr, src1->count());
    return src1;
}

#endif //CUDA_PROJ_TENSOR_CUH
