#pragma once

#include "hypergraph.h"

namespace Eigen {

template <typename T>
struct NumTraits;

template <>
struct NumTraits<hypergraph::Variable<double>> : NumTraits<double> {
    using Real = hypergraph::Variable<double>;
    using NonInteger = hypergraph::Variable<double>;
    using Nested = hypergraph::Variable<double>;
    enum {
        IsComplex = 0,
        IsInteger = 0,
        IsSigned = 1,
        RequireInitialization = 1,
        ReadCost = 1,
        AddCost = 3,
        MulCost = 3
    };
};

template <typename BinOp>
struct ScalarBinaryOpTraits<hypergraph::Variable<double>, double, BinOp> {
    using ReturnType = hypergraph::Variable<double>;
};

template <typename BinOp>
struct ScalarBinaryOpTraits<double, hypergraph::Variable<double>, BinOp> {
    using ReturnType = hypergraph::Variable<double>;
};

#define EIGEN_MAKE_TYPEDEFS(Type, TypeSuffix, Size, SizeSuffix)                     \
    using Matrix##SizeSuffix##TypeSuffix = Matrix<Type, Size, Size, 0, Size, Size>; \
    using Vector##SizeSuffix##TypeSuffix = Matrix<Type, Size, 1, 0, Size, 1>;       \
    using RowVector##SizeSuffix##TypeSuffix = Matrix<Type, 1, Size, 1, 1, Size>;

#define EIGEN_MAKE_FIXED_TYPEDEFS(Type, TypeSuffix, Size)                    \
    using Matrix##Size##X##TypeSuffix = Matrix<Type, Size, -1, 0, Size, -1>; \
    using Matrix##X##Size##TypeSuffix = Matrix<Type, -1, Size, 0, -1, Size>;

#define EIGEN_MAKE_TYPEDEFS_ALL_SIZES(Type, TypeSuffix) \
    EIGEN_MAKE_TYPEDEFS(Type, TypeSuffix, 2, 2)         \
    EIGEN_MAKE_TYPEDEFS(Type, TypeSuffix, 3, 3)         \
    EIGEN_MAKE_TYPEDEFS(Type, TypeSuffix, 4, 4)         \
    EIGEN_MAKE_TYPEDEFS(Type, TypeSuffix, -1, X)        \
    EIGEN_MAKE_FIXED_TYPEDEFS(Type, TypeSuffix, 2)      \
    EIGEN_MAKE_FIXED_TYPEDEFS(Type, TypeSuffix, 3)      \
    EIGEN_MAKE_FIXED_TYPEDEFS(Type, TypeSuffix, 4)

EIGEN_MAKE_TYPEDEFS_ALL_SIZES(hypergraph::Variable<double>, hg)

#undef EIGEN_MAKE_TYPEDEFS_ALL_SIZES
#undef EIGEN_MAKE_TYPEDEFS
#undef EIGEN_MAKE_FIXED_TYPEDEFS

} // namespace Eigen