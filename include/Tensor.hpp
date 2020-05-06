#ifndef CUDA_PROJ_TENSOR_CUH
#define CUDA_PROJ_TENSOR_CUH

#include <stdexcept>
#include <vector>
#include <memory>
#include <iostream>
#include <compute_util.cuh>

typedef std::vector<int> Size;

template<typename T>
class Tensor {

public:
    Tensor(Size size_p);
    virtual ~Tensor();

    int count() const;
    int ndim() const;
    const Size& size();
    void from_cpu(T* ptr);
    void to_cpu(T* ptr);
    Tensor& reshape(const Size& newsize);

    static Tensor* transpose(Tensor* src, Tensor* dst, const std::vector<int>& order);
    Tensor& operator+=(const Tensor& src2);
    Tensor& operator-=(const Tensor& src2);
    Tensor& operator*=(const Tensor& src2);
    Tensor& operator/=(const Tensor& src2);

    T* _ptr;
private:

    Size _size;
    int _count;
    int _ndim;

};


#endif //CUDA_PROJ_TENSOR_CUH
