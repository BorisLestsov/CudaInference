#include "Tensor.hpp"


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
Tensor<T>* Tensor<T>::transpose(Tensor<T>* src, Tensor<T>* dst, const std::vector<int>& order)
{
    // create arrays for reshape
    std::shared_ptr<Tensor<int>> _dims, _reorder, _strides, _new_strides;
    int ndims = src->ndim();

    Size dims_cpu(src->size());
    _dims = std::shared_ptr<Tensor<int>>(new Tensor<int>({ndims}));
    _dims->from_cpu(dims_cpu.data());


    Size strides_cpu(ndims);
    int cnt = 1;
    for (int i = ndims-1; i >= 0; --i){
        strides_cpu[i] = cnt;
        cnt *= dims_cpu[i];
    }
    _strides = std::shared_ptr<Tensor<int>>(new Tensor<int>({ndims}));
    _strides->from_cpu(strides_cpu.data());

    Size reorder_cpu(order);
    _reorder = std::shared_ptr<Tensor<int>>(new Tensor<int>({ndims}));
    _reorder->from_cpu(reorder_cpu.data());

    Size new_strides_cpu(ndims);
    cnt = 1;
    for (int i = ndims-1; i >= 0; --i){
        new_strides_cpu[i] = cnt;
        cnt *= dims_cpu[reorder_cpu[i]];
    }
    _new_strides = std::shared_ptr<Tensor<int>>(new Tensor<int>({ndims}));
    _new_strides->from_cpu(new_strides_cpu.data());

    cuda_transpose(src->_ptr, dst->_ptr, _dims->_ptr, _strides->_ptr, _reorder->_ptr, _new_strides->_ptr, ndims, src->count());

    return dst;
}

template<typename T>
Tensor<T>& Tensor<T>::operator+=(const Tensor<T>& src2)
{
    if (this->count() != src2.count()){
        throw std::runtime_error("different size in Tensor::add_inplace");
    }
    cuda_add(this->_ptr, src2._ptr, this->_ptr, this->count());
    return *this;
}
template<typename T>
Tensor<T>& Tensor<T>::operator-=(const Tensor<T>& src2)
{
    if (this->count() != src2.count()){
        throw std::runtime_error("different size in Tensor::sub_inplace");
    }
    cuda_sub(this->_ptr, src2._ptr, this->_ptr, this->count());
    return *this;
}
template<typename T>
Tensor<T>& Tensor<T>::operator*=(const Tensor<T>& src2)
{
    if (this->count() != src2.count()){
        throw std::runtime_error("different size in Tensor::mul_inplace");
    }
    cuda_mul(this->_ptr, src2._ptr, this->_ptr, this->count());
    return *this;
}
template<typename T>
Tensor<T>& Tensor<T>::operator/=(const Tensor<T>& src2)
{
    if (this->count() != src2.count()){
        throw std::runtime_error("different size in Tensor::div_inplace");
    }
    cuda_div(this->_ptr, src2._ptr, this->_ptr, this->count());
    return *this;
}

template class Tensor<float>;
template class Tensor<int>;
