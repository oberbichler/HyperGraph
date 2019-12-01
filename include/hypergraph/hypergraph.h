#pragma once

#include <Eigen/Core>

#include <tsl/robin_map.h>

#include <algorithm>
#include <cstddef>
#include <memory>

namespace hypergraph {

#if defined(_MSC_VER)
#  define HYPERGRAPH_INLINE                 __forceinline
#else
#  define HYPERGRAPH_INLINE                 __attribute__ ((always_inline)) inline
#endif

using index = std::ptrdiff_t;

template <typename TContainer>
HYPERGRAPH_INLINE index length(const TContainer& container)
{
    return static_cast<index>(container.size());
}

template <typename T>
HYPERGRAPH_INLINE std::pair<T, T> minmax(const T a, const T b)
{
    if (a < b) {
        return {a, b};
    } else {
        return {b, a};
    }
}

template <typename T>
class HyperGraph;

template <typename T>
class Variable;

template <typename T>
class Edge;

template <typename T>
HYPERGRAPH_INLINE Variable<T> abs(const Variable<T>& x);

template <typename T>
HYPERGRAPH_INLINE Variable<T> pow(const Variable<T>& x, const double a);

template <typename T>
HYPERGRAPH_INLINE Variable<T> acos(const Variable<T>& x);

template <typename T>
HYPERGRAPH_INLINE Variable<T> asin(const Variable<T>& x);

template <typename T>
HYPERGRAPH_INLINE Variable<T> atan(const Variable<T>& x);

template <typename T>
HYPERGRAPH_INLINE Variable<T> cos(const Variable<T>& x);

template <typename T>
HYPERGRAPH_INLINE Variable<T> sin(const Variable<T>& x);

template <typename T>
HYPERGRAPH_INLINE Variable<T> sqrt(const Variable<T>& x);

template <typename T>
HYPERGRAPH_INLINE Variable<T> tan(const Variable<T>& x);

template <typename T, typename U>
HYPERGRAPH_INLINE index vertex_id(const U& item)
{
    if constexpr (std::is_same<U, index>::value) {
        return item;
    }
    if constexpr (std::is_same<U, Variable<T>>::value) {
        return item.id();
    }
    if constexpr (std::is_same<U, Edge<T>>::value) {
        return item.to();
    }
}

template <typename T>
class Variable {
private: // types
    using Type = Variable<T>;

private: // variables
    HyperGraph<T>* m_graph;
    index m_id;
    T m_value;

public: // constructor
    Variable()
    {
    }

    Variable(HyperGraph<T>* graph, const T value, const index id)
        : m_graph(graph)
        , m_value(value)
        , m_id(id)
    {
    }

public: // methods
    index id() const
    {
        return m_id;
    }

    HyperGraph<T>* graph() const
    {
        return m_graph;
    }

    T value() const
    {
        return m_value;
    }

    void set_value(const T value)
    {
        m_value = value;
    }

public: // python
    template <typename TModule>
    static void register_python(TModule& m)
    {
        using namespace pybind11::literals;
        namespace py = pybind11;

        const std::string name = "Variable";

        py::class_<Type>(m, name.c_str())
            // properties
            .def_property_readonly("value", &Type::value)
            // read-only properties
            .def_property_readonly("_id", &Type::id)
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
            .def(py::self /= py::self);
    }
};

template <typename T>
class Edge {
private: // types
    using Type = Edge<T>;

private: // variables
    index m_to;
    T m_weight;

public: // constructors
    Edge()
    {
    }

    Edge(const index to, const T w = 0.0)
        : m_to(to)
        , m_weight(w)
    {
    }

public: // methods
    index to() const
    {
        return m_to;
    }

    void set_to(const index value)
    {
        m_to = value;
    }

    T weight() const
    {
        return m_weight;
    }

    void set_weight(const T value)
    {
        m_weight = value;
    }
};

template <typename T>
class Vertex {
private: // types
    using Type = Vertex;

private: // variables
    index m_id;
    Edge<T> m_edge1;
    Edge<T> m_edge2;
    T m_weight;
    T m_second_order_weight;

public: // constructor
    Vertex(const index id)
        : m_id(id)
        , m_edge1(id)
        , m_edge2(id)
        , m_weight(0)
        , m_second_order_weight(0)
    {
    }

public: // methods
    index id() const
    {
        return m_id;
    }

    const Edge<T>& edge1() const
    {
        return m_edge1;
    }

    Edge<T>& edge1()
    {
        return m_edge1;
    }

    const Edge<T>& edge2() const
    {
        return m_edge2;
    }

    Edge<T>& edge2()
    {
        return m_edge2;
    }

    T weight() const
    {
        return m_weight;
    }

    T& weight()
    {
        return m_weight;
    }

    T second_order_weight() const
    {
        return m_second_order_weight;
    }

    T& second_order_weight()
    {
        return m_second_order_weight;
    }
};

template <typename T>
class HyperGraph {
private: // types
    using Type = HyperGraph;

private: // variables
    std::vector<Variable<T>> m_variables;
    std::vector<Vertex<T>> m_vertices;
    std::vector<tsl::robin_map<index, T>> m_second_order_edges;
    std::vector<T> m_self_second_order_edges;

public: // constructor
public: // methods
    Variable<T> new_tmp_variable(const T value)
    {
        const index id = length(m_vertices);
        m_vertices.emplace_back(id);
        return Variable(this, value, id);
    }

    Variable<T> new_variable(const T value)
    {
        const index id = length(m_vertices);
        m_vertices.emplace_back(id);
        return m_variables.emplace_back(this, value, id);
    }

    std::vector<Variable<T>> new_variables(const std::vector<double>& values)
    {
        std::vector<Variable<T>> variables(values.size());
        for (index i = 0; i < length(values); i++) {
            variables[i] = new_variable(values[i]);
        }
        return variables;
    }

    void add_edge(const Variable<T>& c, const Variable<T>& p, const T w, const T second_order_weight)
    {
        Vertex<T>& v = vertex(c);
        v.edge1() = {vertex_id<T>(p), w};
        v.second_order_weight() = second_order_weight;
    }

    void add_edge(const Variable<T>& c, const Variable<T>& p1, const Variable<T>& p2, const T w1, const T w2, const T second_order_weight)
    {
        Vertex<T>& v = vertex(c);
        v.edge1() = {vertex_id<T>(p1), w1};
        v.edge2() = {vertex_id<T>(p2), w2};
        v.second_order_weight() = second_order_weight;
    }

    template <typename U>
    HYPERGRAPH_INLINE Vertex<T> vertex(U a) const
    {
        return m_vertices[vertex_id<T>(a)];
    }

    template <typename U>
    HYPERGRAPH_INLINE Vertex<T>& vertex(U a)
    {
        return m_vertices[vertex_id<T>(a)];
    }

    template <typename U>
    HYPERGRAPH_INLINE T self_second_order_edge(U a) const
    {
        return m_self_second_order_edges[vertex_id<T>(a)];
    }

    template <typename U>
    HYPERGRAPH_INLINE T& self_second_order_edge(U a)
    {
        return m_self_second_order_edges[vertex_id<T>(a)];
    }

    template <typename U>
    HYPERGRAPH_INLINE T second_order_edge(U a, U b) const
    {
        const auto [min, max] = minmax(vertex_id<T>(a), vertex_id<T>(b));

        const auto it = m_second_order_edges[max].find(min);
        if (it == m_second_order_edges[max].end()) {
            return 0.0;
        }
        return it->second;
    }

    template <typename U>
    HYPERGRAPH_INLINE T& second_order_edge(U a, U b)
    {
        const auto [min, max] = minmax(vertex_id<T>(a), vertex_id<T>(b));

        return m_second_order_edges[max][min];
    }

    void set_adjoint(const Variable<T>& variable, const T value)
    {
        vertex(variable).weight() = value;
    }

    T get_adjoint(const Variable<T>& variable) const
    {
        return vertex(variable).weight();
    }

    T get_adjoint(const Variable<T>& variable_i, const Variable<T>& variable_j) const
    {
        if (variable_i.id() == variable_j.id()) {
            return self_second_order_edge(variable_i);
        } else {
            return second_order_edge(variable_i, variable_j);
        }
    }

    void push_edge(const Edge<T>& fo_edge, const Edge<T>& so_edge)
    {
        if (fo_edge.to() == so_edge.to()) {
            self_second_order_edge(fo_edge) += 2.0 * fo_edge.weight() * so_edge.weight();
        } else {
            second_order_edge(fo_edge, so_edge) += fo_edge.weight() * so_edge.weight();
        }
    }

    void clear()
    {
        std::fill(m_self_second_order_edges.begin(), m_self_second_order_edges.end(), 0.0);

        for (auto& tree : m_second_order_edges) {
            tree.clear();
        }

        for (auto& it : m_vertices) {
            it.weight() = 0.0;
        }
    }

    void propagate_adjoint()
    {
        if (length(m_vertices) > length(m_second_order_edges)) {
            m_second_order_edges.resize(m_vertices.size());
        } else {
            for (index i = 0; i < length(m_second_order_edges); i++) {
                m_second_order_edges[i].clear();
            }
        }

        m_self_second_order_edges.resize(m_vertices.size(), 0.0);

        for (index vid = length(m_vertices) - 1; vid > 0; vid--) {
            Vertex<T>& v = vertex(vid);
            Edge<T>& e1 = v.edge1();
            Edge<T>& e2 = v.edge2();

            if (e1.to() == vid) {
                continue;
            }

            auto& btree = m_second_order_edges[vid];

            if (e2.to() == vid) {
                for (const auto it : btree) {
                    Edge<T> so_edge(it.first, it.second);
                    push_edge(e1, so_edge);
                }
            } else {
                for (const auto it : btree) {
                    Edge<T> so_edge(it.first, it.second);
                    push_edge(e1, so_edge);
                    push_edge(e2, so_edge);
                }
            }

            if (self_second_order_edge(vid) != 0.0) {
                self_second_order_edge(e1) += e1.weight() * e1.weight() * self_second_order_edge(vid);

                if (e2.to() != vid) {
                    self_second_order_edge(e2) += e2.weight() * e2.weight() * self_second_order_edge(vid);

                    if (e1.to() == e2.to()) {
                        self_second_order_edge(e2) += 2.0 * e1.weight() * e2.weight() * self_second_order_edge(vid);
                    } else {
                        second_order_edge(e1, e2) += e1.weight() * e2.weight() * self_second_order_edge(vid);
                    }
                }
            }

            const auto a = v.weight();

            if (a != 0.0) {
                if (v.second_order_weight() != 0.0) {
                    if (e2.to() == vid) {
                        self_second_order_edge(e1) += a * v.second_order_weight();
                    } else if (e1.to() == e2.to()) {
                        self_second_order_edge(e1) += 2.0 * a * v.second_order_weight();
                    } else {
                        second_order_edge(e1, e2) += a * v.second_order_weight();
                    }
                }

                v.weight() = 0.0;
                vertex(e1).weight() += a * e1.weight();
                if (e2.to() != vid) {
                    vertex(e2).weight() += a * e2.weight();
                }
            }
        }
    }

    void compute(const Variable<T> expression)
    {
        clear();
        set_adjoint(expression, 1.0);
        propagate_adjoint();
    }

    Eigen::VectorXd g() const
    {
        const index n = length(m_variables);

        Eigen::VectorXd result(n);

        g(result);

        return result;
    }

    void g(Eigen::Ref<Eigen::VectorXd> out) const
    {
        const index n = length(m_variables);

        for (index i = 0; i < n; i++) {
            out(i) = double(get_adjoint(m_variables[i]));
        }
    }

    Eigen::MatrixXd h(const bool full = false) const
    {
        const index n = length(m_variables);

        Eigen::MatrixXd result(n, n);

        h(result, full);

        return result;
    }

    void h(Eigen::Ref<Eigen::MatrixXd> out, const bool full = false) const
    {
        const index n = length(m_variables);

        if (full) {
            for (index i = 0; i < n; i++) {
                for (index j = 0; j < n; j++) {
                    out(i, j) = double(get_adjoint(m_variables[i], m_variables[j]));
                }
            }
        } else {
            for (index i = 0; i < n; i++) {
                for (index j = 0; j < i; j++) {
                    out(i, j) = 0.0;
                }
                for (index j = i; j < n; j++) {
                    out(i, j) = double(get_adjoint(m_variables[i], m_variables[j]));
                }
            }
        }
    }

public: // python
    template <typename TModule>
    static void register_python(TModule& m)
    {
        using namespace pybind11::literals;
        namespace py = pybind11;

        const std::string name = "HyperGraph";

        py::class_<Type>(m, name.c_str())
            // constructors
            .def(py::init<>())
            // methods
            .def("new_variable", &Type::new_variable, "value"_a)
            .def("new_variables", &Type::new_variables, "values"_a)
            .def("compute", &Type::compute, "expression"_a)
            .def("g", py::overload_cast<>(&Type::g, py::const_))
            .def("g", py::overload_cast<Eigen::Ref<Eigen::VectorXd>>(&Type::g, py::const_), "out"_a)
            .def("h", py::overload_cast<const bool>(&Type::h, py::const_), "full"_a = false)
            .def("h", py::overload_cast<Eigen::Ref<Eigen::MatrixXd>, const bool>(&Type::h, py::const_), "out"_a, "full"_a = false);
    }
};

template <typename T>
HYPERGRAPH_INLINE bool operator<(const Variable<T>& lhs, const Variable<T>& rhs)
{
    return lhs.value() < rhs.value();
}

template <typename T>
HYPERGRAPH_INLINE bool operator<(const double lhs, const Variable<T>& rhs)
{
    return lhs < rhs.value();
}

template <typename T>
HYPERGRAPH_INLINE bool operator<(const Variable<T>& lhs, const double rhs)
{
    return lhs.value() < rhs;
}

template <typename T>
HYPERGRAPH_INLINE bool operator<=(const Variable<T>& lhs, const Variable<T>& rhs)
{
    return lhs.value() <= rhs.value();
}

template <typename T>
HYPERGRAPH_INLINE bool operator<=(const double lhs, const Variable<T>& rhs)
{
    return lhs <= rhs.value();
}

template <typename T>
HYPERGRAPH_INLINE bool operator<=(const Variable<T>& lhs, const double rhs)
{
    return lhs.value() <= rhs;
}

template <typename T>
HYPERGRAPH_INLINE bool operator>(const Variable<T>& lhs, const Variable<T>& rhs)
{
    return lhs.value() > rhs.value();
}

template <typename T>
HYPERGRAPH_INLINE bool operator>(const double lhs, const Variable<T>& rhs)
{
    return lhs > rhs.value();
}

template <typename T>
HYPERGRAPH_INLINE bool operator>(const Variable<T>& lhs, const double rhs)
{
    return lhs.value() > rhs;
}

template <typename T>
HYPERGRAPH_INLINE bool operator>=(const Variable<T>& lhs, const Variable<T>& rhs)
{
    return lhs.value() >= rhs.value();
}

template <typename T>
HYPERGRAPH_INLINE bool operator>=(const double lhs, const Variable<T>& rhs)
{
    return lhs >= rhs.value();
}

template <typename T>
HYPERGRAPH_INLINE bool operator>=(const Variable<T>& lhs, const double rhs)
{
    return lhs.value() >= rhs;
}

template <typename T>
HYPERGRAPH_INLINE bool operator==(const Variable<T>& lhs, const Variable<T>& rhs)
{
    return lhs.value() == rhs.value();
}

template <typename T>
HYPERGRAPH_INLINE bool operator==(const double lhs, const Variable<T>& rhs)
{
    return lhs == rhs.value();
}

template <typename T>
HYPERGRAPH_INLINE bool operator==(const Variable<T>& lhs, const double rhs)
{
    return lhs.value() == rhs;
}

template <typename T>
HYPERGRAPH_INLINE bool operator!=(const Variable<T>& lhs, const Variable<T>& rhs)
{
    return lhs.value() != rhs.value();
}

template <typename T>
HYPERGRAPH_INLINE bool operator!=(const double lhs, const Variable<T>& rhs)
{
    return lhs != rhs.value();
}

template <typename T>
HYPERGRAPH_INLINE bool operator!=(const Variable<T>& lhs, const double rhs)
{
    return lhs.value() != rhs;
}

template <typename T>
HYPERGRAPH_INLINE Variable<T> operator-(const Variable<T>& x)
{
    HyperGraph<T>* graph = x.graph();

    const Variable<T> result = graph->new_tmp_variable(-x.value());
    graph->add_edge(result, x, -1.0, 0.0);
    return result;
}

template <typename T>
HYPERGRAPH_INLINE Variable<T> operator+(const Variable<T>& lhs, const Variable<T>& rhs)
{
    HyperGraph<T>* graph = lhs.graph();

    const Variable<T> result = graph->new_tmp_variable(lhs.value() + rhs.value());
    graph->add_edge(result, lhs, rhs, 1.0, 1.0, 0.0);
    return result;
}

template <typename T>
HYPERGRAPH_INLINE Variable<T> operator+(const Variable<T>& lhs, const double rhs)
{
    HyperGraph<T>* graph = lhs.graph();

    const Variable<T> result = graph->new_tmp_variable(lhs.value() + rhs);
    graph->add_edge(result, lhs, 1.0, 0.0);
    return result;
}

template <typename T>
HYPERGRAPH_INLINE Variable<T> operator+(const double lhs, const Variable<T>& rhs)
{
    return rhs + lhs;
}

template <typename T>
HYPERGRAPH_INLINE Variable<T>& operator+=(Variable<T>& lhs, const Variable<T>& rhs)
{
    lhs = lhs + rhs;
    return lhs;
}

template <typename T>
HYPERGRAPH_INLINE Variable<T>& operator+=(Variable<T>& lhs, const double rhs)
{
    lhs = lhs + rhs;
    return lhs;
}

template <typename T>
HYPERGRAPH_INLINE Variable<T> operator-(const Variable<T>& lhs, const Variable<T>& rhs)
{
    HyperGraph<T>* graph = lhs.graph();

    const Variable<T> result = graph->new_tmp_variable(lhs.value() - rhs.value());
    graph->add_edge(result, lhs, rhs, 1.0, -1.0, 0.0);
    return result;
}

template <typename T>
HYPERGRAPH_INLINE Variable<T> operator-(const Variable<T>& lhs, const double rhs)
{
    HyperGraph<T>* graph = lhs.graph();

    const Variable<T> result = graph->new_tmp_variable(lhs.value() - rhs);
    graph->add_edge(result, lhs, 1.0, 0.0);
    return result;
}

template <typename T>
HYPERGRAPH_INLINE Variable<T> operator-(const double lhs, const Variable<T>& rhs)
{
    HyperGraph<T>* graph = rhs.graph();

    const Variable<T> result = graph->new_tmp_variable(lhs - rhs.value());
    graph->add_edge(result, rhs, -1.0, 0.0);
    return result;
}

template <typename T>
HYPERGRAPH_INLINE Variable<T>& operator-=(Variable<T>& lhs, const Variable<T>& rhs)
{
    lhs = lhs - rhs;
    return lhs;
}

template <typename T>
HYPERGRAPH_INLINE Variable<T>& operator-=(Variable<T>& lhs, const double rhs)
{
    lhs = lhs - rhs;
    return lhs;
}

template <typename T>
HYPERGRAPH_INLINE Variable<T> operator*(const Variable<T>& lhs, const Variable<T>& rhs)
{
    HyperGraph<T>* graph = lhs.graph();

    const Variable<T> result = graph->new_tmp_variable(lhs.value() * rhs.value());
    graph->add_edge(result, lhs, rhs, rhs.value(), lhs.value(), 1.0);
    return result;
}

template <typename T>
HYPERGRAPH_INLINE Variable<T> operator*(const Variable<T>& lhs, const double rhs)
{
    HyperGraph<T>* graph = lhs.graph();

    const Variable<T> result = graph->new_tmp_variable(lhs.value() * rhs);
    graph->add_edge(result, lhs, rhs, 0.0);
    return result;
}

template <typename T>
HYPERGRAPH_INLINE Variable<T> operator*(const double lhs, const Variable<T>& rhs)
{
    return rhs * lhs;
}

template <typename T>
HYPERGRAPH_INLINE Variable<T>& operator*=(Variable<T>& lhs, const Variable<T>& rhs)
{
    lhs = lhs * rhs;
    return lhs;
}

template <typename T>
HYPERGRAPH_INLINE Variable<T>& operator*=(Variable<T>& lhs, const double rhs)
{
    lhs = lhs * rhs;
    return lhs;
}

template <typename T>
HYPERGRAPH_INLINE Variable<T> inv(const Variable<T>& x)
{
    HyperGraph<T>* graph = x.graph();

    const auto inv_x = 1.0 / x.value();
    const auto inv_x_sq = inv_x * inv_x;
    const auto inv_x_cu = inv_x_sq * inv_x;
    const Variable<T> result = graph->new_tmp_variable(inv_x);
    graph->add_edge(result, x, -inv_x_sq, 2.0 * inv_x_cu);
    return result;
}

template <typename T>
HYPERGRAPH_INLINE Variable<T> operator/(const Variable<T>& lhs, const Variable<T>& rhs)
{
    return lhs * inv(rhs);
}

template <typename T>
HYPERGRAPH_INLINE Variable<T> operator/(const Variable<T>& lhs, const double rhs)
{
    const auto inv = 1.0 / rhs;
    return lhs * inv;
}

template <typename T>
HYPERGRAPH_INLINE Variable<T> operator/(const double lhs, const Variable<T>& rhs)
{
    return lhs * inv(rhs);
}

template <typename T>
HYPERGRAPH_INLINE Variable<T>& operator/=(Variable<T>& lhs, const Variable<T>& rhs)
{
    lhs = lhs / rhs;
    return lhs;
}

template <typename T>
HYPERGRAPH_INLINE Variable<T>& operator/=(Variable<T>& lhs, const double rhs)
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

template <typename T>
HYPERGRAPH_INLINE Variable<T> abs(const Variable<T>& x)
{
    if (x.value() > 0) {
        return x;
    } else {
        return -x;
    }
}

template <typename T>
HYPERGRAPH_INLINE Variable<T> square(const Variable<T>& x)
{
    HyperGraph<T>* graph = x.graph();

    const auto sq = x.value() * x.value();
    const Variable<T> result = graph->new_tmp_variable(sq);
    graph->add_edge(result, x, 2.0 * x.value(), 0.0);
    return result;
}

template <typename T>
HYPERGRAPH_INLINE Variable<T> sqrt(const Variable<T>& x)
{
    using std::sqrt;

    HyperGraph<T>* graph = x.graph();

    const auto sqrt_x = sqrt(x.value());
    const auto inv_sqrt = 1.0 / sqrt_x;
    const Variable<T> result = graph->new_tmp_variable(sqrt_x);
    graph->add_edge(result, x, 0.5 * inv_sqrt, -0.25 * inv_sqrt / x.value());
    return result;
}

template <typename T>
HYPERGRAPH_INLINE Variable<T> pow(const Variable<T>& x, const double a)
{
    using std::pow;

    HyperGraph<T>* graph = x.graph();

    const auto pow_x = pow(x.value(), a);
    const Variable<T> result = graph->new_tmp_variable(pow_x);
    graph->add_edge(result, x, a * pow(x.value(), a - 1.0), a * (a - 1.0) * pow(x.value(), a - 2.0));
    return result;
}

template <typename T>
HYPERGRAPH_INLINE Variable<T> exp(const Variable<T>& x)
{
    using std::exp;

    HyperGraph<T>* graph = x.graph();

    const auto exp_x = exp(x.value());
    const Variable<T> result = graph->new_tmp_variable(exp_x);
    graph->add_edge(result, x, exp_x, exp_x);
    return result;
}

template <typename T>
HYPERGRAPH_INLINE Variable<T> log(const Variable<T>& x)
{
    using std::log;

    HyperGraph<T>* graph = x.graph();

    const auto log_x = log(x.value());
    const Variable<T> result = graph->new_tmp_variable(log_x);
    const auto inv = 1.0 / x.value();
    graph->add_edge(result, x, inv, -inv * inv);
    return result;
}

template <typename T>
HYPERGRAPH_INLINE Variable<T> cos(const Variable<T>& x)
{
    using std::cos;
    using std::sin;

    HyperGraph<T>* graph = x.graph();

    const auto cos_x = cos(x.value());
    const Variable<T> result = graph->new_tmp_variable(cos_x);
    graph->add_edge(result, x, -sin(x.value()), -cos_x);
    return result;
}

template <typename T>
HYPERGRAPH_INLINE Variable<T> sin(const Variable<T>& x)
{
    using std::cos;
    using std::sin;

    HyperGraph<T>* graph = x.graph();

    const auto sin_x = sin(x.value());
    const Variable<T> result = graph->new_tmp_variable(sin_x);
    graph->add_edge(result, x, cos(x.value()), -sin_x);
    return result;
}

template <typename T>
HYPERGRAPH_INLINE Variable<T> tan(const Variable<T>& x)
{
    using std::tan;
    using std::cos;

    HyperGraph<T>* graph = x.graph();

    const auto tan_x = tan(x.value());
    const auto sec = 1.0 / cos(x.value());
    const auto sec_sq = sec * sec;
    const Variable<T> result = graph->new_tmp_variable(tan_x);
    graph->add_edge(result, x, sec_sq, 2.0 * tan_x * sec_sq);
    return result;
}

template <typename T>
HYPERGRAPH_INLINE Variable<T> acos(const Variable<T>& x)
{
    using std::acos;
    using std::sqrt;

    HyperGraph<T>* graph = x.graph();

    const auto acos_x = acos(x.value());
    const Variable<T> result = graph->new_tmp_variable(acos_x);
    const auto tmp = 1.0 / (1.0 - x.value() * x.value());
    const auto neg_tmp_sqrt = -sqrt(tmp);
    graph->add_edge(result, x, neg_tmp_sqrt, x.value() * neg_tmp_sqrt * tmp);
    return result;
}

template <typename T>
HYPERGRAPH_INLINE Variable<T> asin(const Variable<T>& x)
{
    using std::asin;
    using std::sqrt;

    HyperGraph<T>* graph = x.graph();

    const auto asin_x = asin(x.value());
    const Variable<T> result = graph->new_tmp_variable(asin_x);
    const auto tmp = 1.0 / (1.0 - x.value() * x.value());
    const auto tmp_sqrt = sqrt(tmp);
    graph->add_edge(result, x, tmp_sqrt, x.value() * tmp_sqrt * tmp);
    return result;
}

template <typename T>
HYPERGRAPH_INLINE Variable<T> atan(const Variable<T>& x)
{
    using std::atan;

    HyperGraph<T>* graph = x.graph();

    const auto atan_x = atan(x.value());
    const Variable<T> result = graph->new_tmp_variable(atan_x);
    const auto tmp = 1 / (1 + x.value() * x.value());
    graph->add_edge(result, x, tmp, -2 * x.value() * tmp * tmp);
    return result;
}

} // namespace hypergraph
