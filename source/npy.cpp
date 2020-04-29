
#include "npy.hpp"

namespace npy {
constexpr dtype_t has_typestring<float>::dtype;
constexpr dtype_t has_typestring<double>::dtype;
constexpr dtype_t has_typestring<long double>::dtype;
constexpr dtype_t has_typestring<char>::dtype;
constexpr dtype_t has_typestring<short>::dtype;
constexpr dtype_t has_typestring<int>::dtype;
constexpr dtype_t has_typestring<long>::dtype;
constexpr dtype_t has_typestring<long long>::dtype;
constexpr dtype_t has_typestring<unsigned char>::dtype;
constexpr dtype_t has_typestring<unsigned short>::dtype;
constexpr dtype_t has_typestring<unsigned int>::dtype;
constexpr dtype_t has_typestring<unsigned long>::dtype;
constexpr dtype_t has_typestring<unsigned long long>::dtype;
constexpr dtype_t has_typestring<std::complex<float>>::dtype;
constexpr dtype_t has_typestring<std::complex<double>>::dtype;
constexpr dtype_t has_typestring<std::complex<long double>>::dtype;
}
