#pragma once

#include <algorithm>

namespace hypergraph {

class Counting;

inline Counting abs(const Counting& x);
inline Counting pow(const Counting& x, const double a);
inline Counting acos(const Counting& x);
inline Counting asin(const Counting& x);
inline Counting atan(const Counting& x);
inline Counting cos(const Counting& x);
inline Counting sin(const Counting& x);
inline Counting sqrt(const Counting& x);
inline Counting tan(const Counting& x);

class Counting
{
private:
    using Type = Counting;

private:
    double m_value;

public:
    static inline int m_nb_add = 0;
    static inline int m_nb_sub = 0;
    static inline int m_nb_mul = 0;
    static inline int m_nb_div = 0;
    static inline int m_nb_sqrt = 0;

public:
    Counting() : m_value(0.0)
    {
    }

    Counting(double value) : m_value(value)
    {
    }
    
public:
    double value() const
    {
        return m_value;
    }
    
    double& value()
    {
        return m_value;
    }
    
    void set_value(const double value)
    {
        m_value = value;
    }

    explicit operator double() const { return m_value; }

public: // python
    template <typename TModule>
    static void register_python(TModule& m)
    {
        using namespace pybind11::literals;
        namespace py = pybind11;

        const std::string name = "Counting";

        py::class_<Type>(m, name.c_str())
            // constructors
            .def(py::init<double>())
            // properties
            .def_property("value", py::overload_cast<>(&Type::value, py::const_), &Type::set_value)
            // methods
            .def("__abs__", [](const Type& x) { return hypergraph::abs(x); })
            .def("__pow__", [](const Type& x, const double a) { return hypergraph::pow(x, a); })
            .def("arccos", [](const Type& x) { return hypergraph::acos(x); })
            .def("arcsin", [](const Type& x) { return hypergraph::asin(x); })
            .def("arctan", [](const Type& x) { return hypergraph::atan(x); })
            .def("cos", [](const Type& x) { return hypergraph::cos(x); })
            .def("sin", [](const Type& x) { return hypergraph::sin(x); })
            .def("sqrt", [](const Type& x) { return hypergraph::sqrt(x); })
            .def("tan", [](const Type& x) { return hypergraph::tan(x); })
            .def_static("clear", []() {
                Type::m_nb_add = 0;
                Type::m_nb_sub = 0;
                Type::m_nb_mul = 0;
                Type::m_nb_div = 0;
                Type::m_nb_sqrt = 0;
            })
            // operators
            .def(py::self == py::self)
            .def(py::self != py::self)
            .def(py::self < py::self)
            .def(py::self > py::self)
            .def(py::self <= py::self)
            .def(py::self >= py::self)
            .def(py::self == double())
            .def(py::self != double())
            .def(py::self < double())
            .def(py::self > double())
            .def(py::self <= double())
            .def(py::self >= double())
            .def(double() == py::self)
            .def(double() != py::self)
            .def(double() < py::self)
            .def(double() > py::self)
            .def(double() <= py::self)
            .def(double() >= py::self)
            .def(-py::self)
            .def(py::self + py::self)
            .def(py::self + double())
            .def(double() + py::self)
            .def(py::self += py::self)
            .def(py::self - py::self)
            .def(py::self - double())
            .def(double() - py::self)
            .def(py::self -= py::self)
            .def(py::self * py::self)
            .def(py::self * double())
            .def(double() * py::self)
            .def(py::self *= py::self)
            .def(py::self / py::self)
            .def(py::self / double())
            .def(double() / py::self)
            .def(py::self /= py::self)
            .def_property_readonly_static("nb_add", [](const py::object& self) { return Counting::m_nb_add; })
            .def_property_readonly_static("nb_sub", [](const py::object& self) { return Counting::m_nb_sub; })
            .def_property_readonly_static("nb_mul", [](const py::object& self) { return Counting::m_nb_mul; })
            .def_property_readonly_static("nb_div", [](const py::object& self) { return Counting::m_nb_div; })
            .def_property_readonly_static("nb_sqrt", [](const py::object& self) { return Counting::m_nb_sqrt; })
            ;
    }
};

inline bool operator<(const Counting& lhs, const Counting& rhs)
{
    return lhs.value() < rhs.value();
}

inline bool operator<(const double lhs, const Counting& rhs)
{
    return lhs < rhs.value();
}

inline bool operator<(const Counting& lhs, const double rhs)
{
    return lhs.value() < rhs;
}

inline bool operator<=(const Counting& lhs, const Counting& rhs)
{
    return lhs.value() <= rhs.value();
}

inline bool operator<=(const double lhs, const Counting& rhs)
{
    return lhs <= rhs.value();
}

inline bool operator<=(const Counting& lhs, const double rhs)
{
    return lhs.value() <= rhs;
}

inline bool operator>(const Counting& lhs, const Counting& rhs)
{
    return lhs.value() > rhs.value();
}

inline bool operator>(const double lhs, const Counting& rhs)
{
    return lhs > rhs.value();
}

inline bool operator>(const Counting& lhs, const double rhs)
{
    return lhs.value() > rhs;
}

inline bool operator>=(const Counting& lhs, const Counting& rhs)
{
    return lhs.value() >= rhs.value();
}

inline bool operator>=(const double lhs, const Counting& rhs)
{
    return lhs >= rhs.value();
}

inline bool operator>=(const Counting& lhs, const double rhs)
{
    return lhs.value() >= rhs;
}

inline bool operator==(const Counting& lhs, const Counting& rhs)
{
    return lhs.value() == rhs.value();
}

inline bool operator==(const double lhs, const Counting& rhs)
{
    return lhs == rhs.value();
}

inline bool operator==(const Counting& lhs, const double rhs)
{
    return lhs.value() == rhs;
}

inline bool operator!=(const Counting& lhs, const Counting& rhs)
{
    return lhs.value() != rhs.value();
}

inline bool operator!=(const double lhs, const Counting& rhs)
{
    return lhs != rhs.value();
}

inline bool operator!=(const Counting& lhs, const double rhs)
{
    return lhs.value() != rhs;
}

inline Counting operator-(const Counting& x)
{
    Counting::m_nb_sub += 1;
    return Counting(-x.value());
}

inline Counting operator+(const Counting& lhs, const Counting& rhs)
{
    Counting::m_nb_add += 1;
    return Counting(lhs.value() + rhs.value());
}

inline Counting operator+(const Counting& lhs, const double rhs)
{
    Counting::m_nb_add += 1;
    return Counting(lhs.value() + rhs);
}

inline Counting operator+(const double lhs, const Counting& rhs)
{
    Counting::m_nb_add += 1;
    return Counting(lhs + rhs.value());
}

inline Counting& operator+=(Counting& lhs, const Counting& rhs)
{
    lhs = lhs + rhs;
    return lhs;
}

inline Counting& operator+=(Counting& lhs, const double rhs)
{
    lhs = lhs + rhs;
    return lhs;
}

inline Counting operator-(const Counting& lhs, const Counting& rhs)
{
    Counting::m_nb_sub += 1;
    return Counting(lhs.value() - rhs.value());
}

inline Counting operator-(const Counting& lhs, const double rhs)
{
    Counting::m_nb_sub += 1;
    return Counting(lhs.value() - rhs);
}

inline Counting operator-(const double lhs, const Counting& rhs)
{
    Counting::m_nb_sub += 1;
    return Counting(lhs - rhs.value());
}

inline Counting& operator-=(Counting& lhs, const Counting& rhs)
{
    lhs = lhs - rhs;
    return lhs;
}

inline Counting& operator-=(Counting& lhs, const double rhs)
{
    lhs = lhs - rhs;
    return lhs;
}

inline Counting operator*(const Counting& lhs, const Counting& rhs)
{
    Counting::m_nb_mul += 1;
    return Counting(lhs.value() * rhs.value());
}

inline Counting operator*(const Counting& lhs, const double rhs)
{
    Counting::m_nb_mul += 1;
    return Counting(lhs.value() * rhs);
}

inline Counting operator*(const double lhs, const Counting& rhs)
{
    Counting::m_nb_mul += 1;
    return Counting(lhs * rhs.value());
}

inline Counting& operator*=(Counting& lhs, const Counting& rhs)
{
    lhs = lhs * rhs;
    return lhs;
}

inline Counting& operator*=(Counting& lhs, const double rhs)
{
    lhs = lhs * rhs;
    return lhs;
}

inline Counting operator/(const Counting& lhs, const Counting& rhs)
{
    Counting::m_nb_div += 1;
    return Counting(lhs.value() / rhs.value());
}

inline Counting operator/(const Counting& lhs, const double rhs)
{
    Counting::m_nb_div += 1;
    return Counting(lhs.value() / rhs);
}

inline Counting operator/(const double lhs, const Counting& rhs)
{
    Counting::m_nb_div += 1;
    return Counting(lhs / rhs.value());
}

inline Counting& operator/=(Counting& lhs, const Counting& rhs)
{
    lhs = lhs / rhs;
    return lhs;
}

inline Counting& operator/=(Counting& lhs, const double rhs)
{
    lhs = lhs / rhs;
    return lhs;
}

using std::abs;
using std::acos;
using std::asin;
using std::atan;
using std::cos;
using std::exp;
using std::log;
using std::pow;
using std::sin;
using std::sqrt;
using std::tan;

inline Counting abs(const Counting& x)
{
    if (x.value() > 0) {
        return x;
    } else {
        return -x;
    }
}

inline Counting square(const Counting& x)
{
    return Counting(x.value() * x.value());
}

inline Counting sqrt(const Counting& x)
{
    Counting::m_nb_sqrt += 1;
    return Counting(std::sqrt(x.value()));
}

inline Counting pow(const Counting& x, const double a)
{
    return Counting(std::pow(x.value(), a));
}

inline Counting exp(const Counting& x)
{
    return Counting(std::exp(x.value()));
}

inline Counting log(const Counting& x)
{
    return Counting(std::log(x.value()));
}

inline Counting cos(const Counting& x)
{
    return Counting(std::cos(x.value()));
}

inline Counting sin(const Counting& x)
{
    return Counting(std::sin(x.value()));
}

inline Counting tan(const Counting& x)
{
    return Counting(std::tan(x.value()));
}

inline Counting acos(const Counting& x)
{
    return Counting(std::acos(x.value()));
}

inline Counting asin(const Counting& x)
{
    return Counting(std::asin(x.value()));
}

inline Counting atan(const Counting& x)
{
    return Counting(std::atan(x.value()));
}

} // namespace hypergraph