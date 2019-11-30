#pragma once

#include <Eigen/Core>

#include <tsl/robin_map.h>

#include <algorithm>
#include <cstddef>
#include <memory>

namespace hypergraph {

using index = std::ptrdiff_t;

template <typename TContainer>
index length(const TContainer& container)
{
    return static_cast<index>(container.size());
}

class HyperGraph;
class Variable;
class Edge;

inline Variable abs(const Variable& x);
inline Variable pow(const Variable& x, const double a);
inline Variable acos(const Variable& x);
inline Variable asin(const Variable& x);
inline Variable atan(const Variable& x);
inline Variable cos(const Variable& x);
inline Variable sin(const Variable& x);
inline Variable sqrt(const Variable& x);
inline Variable tan(const Variable& x);

template <typename T>
inline index vertex_id(const T& item)
{
    if constexpr (std::is_same<T, index>::value) {
        return item;
    }
    if constexpr (std::is_same<T, Variable>::value) {
        return item.id();
    }
    if constexpr (std::is_same<T, Edge>::value) {
        return item.to();
    }
}

class Variable {
private: // info
    using Type = Variable;
    inline static const std::string s_name = "Variable";

private: // variables
    HyperGraph* m_graph;
    index m_id;
    double m_value;

public: // constructor
    Variable()
    {
    }

    Variable(HyperGraph* graph, const double value, const index id)
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

    HyperGraph* graph() const
    {
        return m_graph;
    }

    double value() const
    {
        return m_value;
    }

    void set_value(const double value)
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
            .def("__abs__", [](const Variable& x) { return hypergraph::abs(x); })
            .def("__pow__", [](const Variable& x, const double a) { return hypergraph::pow(x, a); })
            .def("arccos", [](const Variable& x) { return hypergraph::acos(x); })
            .def("arcsin", [](const Variable& x) { return hypergraph::asin(x); })
            .def("arctan", [](const Variable& x) { return hypergraph::atan(x); })
            .def("cos", [](const Variable& x) { return hypergraph::cos(x); })
            .def("sin", [](const Variable& x) { return hypergraph::sin(x); })
            .def("sqrt", [](const Variable& x) { return hypergraph::sqrt(x); })
            .def("tan", [](const Variable& x) { return hypergraph::tan(x); })
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

class Edge {
private: // types
    using Type = Edge;

private: // variables
    index m_to;
    double m_weight;

public: // constructors
    Edge()
    {
    }

    Edge(const index to, const double w = 0.0)
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

    double weight() const
    {
        return m_weight;
    }

    void set_weight(const double value)
    {
        m_weight = value;
    }
};

class Vertex {
private: // types
    using Type = Vertex;

private: // variables
    index m_id;
    Edge m_edge1;
    Edge m_edge2;
    double m_weight;
    double m_second_order_weight;

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

    const Edge& edge1() const
    {
        return m_edge1;
    }

    void set_edge1(const Edge value)
    {
        m_edge1 = value;
    }

    template <typename T>
    void set_edge1(const T to, const double weight)
    {
        m_edge1 = {vertex_id(to), weight};
    }

    Edge& edge1()
    {
        return m_edge1;
    }

    const Edge& edge2() const
    {
        return m_edge2;
    }

    void set_edge2(const Edge value)
    {
        m_edge2 = value;
    }

    template <typename T>
    void set_edge2(const T to, const double weight)
    {
        m_edge2 = {vertex_id(to), weight};
    }

    Edge& edge2()
    {
        return m_edge2;
    }

    double weight() const
    {
        return m_weight;
    }

    void set_weight(const double value)
    {
        m_weight = value;
    }

    double& weight()
    {
        return m_weight;
    }

    double second_order_weight() const
    {
        return m_second_order_weight;
    }

    void set_second_order_weight(const double value)
    {
        m_second_order_weight = value;
    }
};

class HyperGraph {
private: // types
    using Type = HyperGraph;

private: // variables
    std::vector<Variable> m_variables;
    std::vector<Vertex> m_vertices;
    std::vector<tsl::robin_map<index, double>> m_second_order_edges;
    std::vector<double> m_self_second_order_edges;

public: // constructor
public: // methods
    Variable new_tmp_variable(const double value)
    {
        const index id = length(m_vertices);
        m_vertices.emplace_back(id);
        return Variable(this, value, id);
    }

    Variable new_variable(const double value)
    {
        const index id = length(m_vertices);
        m_vertices.emplace_back(id);
        Variable variable(this, value, id);
        m_variables.push_back(variable);
        return variable;
    }

    std::vector<Variable> new_variables(const std::vector<double>& values)
    {
        std::vector<Variable> variables(values.size());
        for (index i = 0; i < length(values); i++) {
            variables[i] = new_variable(values[i]);
        }
        return variables;
    }

    void add_edge(const Variable& c, const Variable& p, const double w, const double second_order_weight)
    {
        Vertex& v = vertex(c);
        v.set_edge1(p, w);
        v.set_second_order_weight(second_order_weight);
    }

    void add_edge(const Variable& c, const Variable& p1, const Variable& p2, const double w1, const double w2, const double second_order_weight)
    {
        Vertex& v = vertex(c);
        v.set_edge1(p1, w1);
        v.set_edge2(p2, w2);
        v.set_second_order_weight(second_order_weight);
    }

    template <typename T>
    inline Vertex vertex(T a) const
    {
        return m_vertices[vertex_id(a)];
    }

    template <typename T>
    inline Vertex& vertex(T a)
    {
        return m_vertices[vertex_id(a)];
    }

    template <typename T>
    inline double self_second_order_edge(T a) const
    {
        return m_self_second_order_edges[vertex_id(a)];
    }

    template <typename T>
    inline double& self_second_order_edge(T a)
    {
        return m_self_second_order_edges[vertex_id(a)];
    }

    template <typename T>
    inline double second_order_edge(T a, T b) const
    {
        index min = vertex_id(a);
        index max = vertex_id(b);

        if (min > max) {
            std::swap(min, max);
        }

        const auto it = m_second_order_edges[max].find(min);
        if (it == m_second_order_edges[max].end()) {
            return 0.0;
        }
        return it->second;
    }

    template <typename T>
    inline double& second_order_edge(T a, T b)
    {
        index min = vertex_id(a);
        index max = vertex_id(b);

        if (min > max) {
            std::swap(min, max);
        }

        return m_second_order_edges[max][min];
    }

    void set_adjoint(const Variable& v, const double adj)
    {
        vertex(v).set_weight(adj);
    }

    double get_adjoint(const Variable& v) const
    {
        return vertex(v).weight();
    }

    double get_adjoint(const Variable& i, const Variable& j) const
    {
        if (i.id() == j.id()) {
            return self_second_order_edge(i);
        } else {
            return second_order_edge(i, j);
        }
    }

    void push_edge(const Edge& fo_edge, const Edge& so_edge)
    {
        if (fo_edge.to() == so_edge.to()) {
            self_second_order_edge(fo_edge) += 2 * fo_edge.weight() * so_edge.weight();
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
            it.set_weight(0.0);
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
            Vertex& v = vertex(vid);
            Edge& e1 = v.edge1();
            Edge& e2 = v.edge2();

            if (e1.to() == vid) {
                continue;
            }

            auto& btree = m_second_order_edges[vid];

            if (e2.to() == vid) {
                for (const auto it : btree) {
                    Edge so_edge(it.first, it.second);
                    push_edge(e1, so_edge);
                }
            } else {
                for (const auto it : btree) {
                    Edge so_edge(it.first, it.second);
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

            const double a = v.weight();

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

    void compute(const Variable expression)
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
            out(i) = get_adjoint(m_variables[i]);
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
                    out(i, j) = get_adjoint(m_variables[i], m_variables[j]);
                }
            }
        } else {
            for (index i = 0; i < n; i++) {
                for (index j = 0; j < i; j++) {
                    out(i, j) = 0.0;
                }
                for (index j = i; j < n; j++) {
                    out(i, j) = get_adjoint(m_variables[i], m_variables[j]);
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

inline bool operator<(const Variable& lhs, const Variable& rhs)
{
    return lhs.value() < rhs.value();
}

inline bool operator<(const double lhs, const Variable& rhs)
{
    return lhs < rhs.value();
}

inline bool operator<(const Variable& lhs, const double rhs)
{
    return lhs.value() < rhs;
}

inline bool operator<=(const Variable& lhs, const Variable& rhs)
{
    return lhs.value() <= rhs.value();
}

inline bool operator<=(const double lhs, const Variable& rhs)
{
    return lhs <= rhs.value();
}

inline bool operator<=(const Variable& lhs, const double rhs)
{
    return lhs.value() <= rhs;
}

inline bool operator>(const Variable& lhs, const Variable& rhs)
{
    return lhs.value() > rhs.value();
}

inline bool operator>(const double lhs, const Variable& rhs)
{
    return lhs > rhs.value();
}

inline bool operator>(const Variable& lhs, const double rhs)
{
    return lhs.value() > rhs;
}

inline bool operator>=(const Variable& lhs, const Variable& rhs)
{
    return lhs.value() >= rhs.value();
}

inline bool operator>=(const double lhs, const Variable& rhs)
{
    return lhs >= rhs.value();
}

inline bool operator>=(const Variable& lhs, const double rhs)
{
    return lhs.value() >= rhs;
}

inline bool operator==(const Variable& lhs, const Variable& rhs)
{
    return lhs.value() == rhs.value();
}

inline bool operator==(const double lhs, const Variable& rhs)
{
    return lhs == rhs.value();
}

inline bool operator==(const Variable& lhs, const double rhs)
{
    return lhs.value() == rhs;
}

inline bool operator!=(const Variable& lhs, const Variable& rhs)
{
    return lhs.value() != rhs.value();
}

inline bool operator!=(const double lhs, const Variable& rhs)
{
    return lhs != rhs.value();
}

inline bool operator!=(const Variable& lhs, const double rhs)
{
    return lhs.value() != rhs;
}

inline Variable operator-(const Variable& x)
{
    HyperGraph* graph = x.graph();

    const Variable result = graph->new_tmp_variable(-x.value());
    graph->add_edge(result, x, -1.0, 0.0);
    return result;
}

inline Variable operator+(const Variable& lhs, const Variable& rhs)
{
    HyperGraph* graph = lhs.graph();

    const Variable result = graph->new_tmp_variable(lhs.value() + rhs.value());
    graph->add_edge(result, lhs, rhs, 1.0, 1.0, 0.0);
    return result;
}

inline Variable operator+(const Variable& lhs, const double rhs)
{
    HyperGraph* graph = lhs.graph();

    const Variable result = graph->new_tmp_variable(lhs.value() + rhs);
    graph->add_edge(result, lhs, 1.0, 0.0);
    return result;
}

inline Variable operator+(const double lhs, const Variable& rhs)
{
    return rhs + lhs;
}

inline Variable& operator+=(Variable& lhs, const Variable& rhs)
{
    lhs = lhs + rhs;
    return lhs;
}

inline Variable& operator+=(Variable& lhs, const double rhs)
{
    lhs = lhs + rhs;
    return lhs;
}

inline Variable operator-(const Variable& lhs, const Variable& rhs)
{
    HyperGraph* graph = lhs.graph();

    const Variable result = graph->new_tmp_variable(lhs.value() - rhs.value());
    graph->add_edge(result, lhs, rhs, 1.0, -1.0, 0.0);
    return result;
}

inline Variable operator-(const Variable& lhs, const double rhs)
{
    HyperGraph* graph = lhs.graph();

    const Variable result = graph->new_tmp_variable(lhs.value() - rhs);
    graph->add_edge(result, lhs, 1.0, 0.0);
    return result;
}

inline Variable operator-(const double lhs, const Variable& rhs)
{
    HyperGraph* graph = rhs.graph();

    const Variable result = graph->new_tmp_variable(lhs - rhs.value());
    graph->add_edge(result, rhs, double(-1.0), 0.0);
    return result;
}

inline Variable& operator-=(Variable& lhs, const Variable& rhs)
{
    lhs = lhs - rhs;
    return lhs;
}

inline Variable& operator-=(Variable& lhs, const double rhs)
{
    lhs = lhs - rhs;
    return lhs;
}

inline Variable operator*(const Variable& lhs, const Variable& rhs)
{
    HyperGraph* graph = lhs.graph();

    const Variable result = graph->new_tmp_variable(lhs.value() * rhs.value());
    graph->add_edge(result, lhs, rhs, rhs.value(), lhs.value(), 1.0);
    return result;
}

inline Variable operator*(const Variable& lhs, const double rhs)
{
    HyperGraph* graph = lhs.graph();

    const Variable result = graph->new_tmp_variable(lhs.value() * rhs);
    graph->add_edge(result, lhs, rhs, 0.0);
    return result;
}

inline Variable operator*(const double lhs, const Variable& rhs)
{
    return rhs * lhs;
}

inline Variable& operator*=(Variable& lhs, const Variable& rhs)
{
    lhs = lhs * rhs;
    return lhs;
}

inline Variable& operator*=(Variable& lhs, const double rhs)
{
    lhs = lhs * rhs;
    return lhs;
}

inline Variable inv(const Variable& x)
{
    HyperGraph* graph = x.graph();

    const double inv_x = 1.0 / x.value();
    const double inv_x_sq = inv_x * inv_x;
    const double inv_x_cu = inv_x_sq * inv_x;
    const Variable result = graph->new_tmp_variable(inv_x);
    graph->add_edge(result, x, -inv_x_sq, 2.0 * inv_x_cu);
    return result;
}

inline double inv(const double x)
{
    return 1.0 / x;
}

inline Variable operator/(const Variable& lhs, const Variable& rhs)
{
    return lhs * inv(rhs);
}

inline Variable operator/(const Variable& lhs, const double rhs)
{
    return lhs * inv(rhs);
}

inline Variable operator/(const double lhs, const Variable& rhs)
{
    return lhs * inv(rhs);
}

inline Variable& operator/=(Variable& lhs, const Variable& rhs)
{
    lhs = lhs / rhs;
    return lhs;
}

inline Variable& operator/=(Variable& lhs, const double rhs)
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

inline Variable abs(const Variable& x)
{
    if (x.value() > 0) {
        return x;
    } else {
        return -x;
    }
}

inline Variable square(const Variable& x)
{
    HyperGraph* graph = x.graph();

    const double sq = x.value() * x.value();
    const Variable result = graph->new_tmp_variable(sq);
    graph->add_edge(result, x, 2.0 * x.value(), 0.0);
    return result;
}

inline Variable sqrt(const Variable& x)
{
    HyperGraph* graph = x.graph();

    const double sqrt = std::sqrt(x.value());
    const double inv_sqrt = 1.0 / sqrt;
    const Variable result = graph->new_tmp_variable(sqrt);
    graph->add_edge(result, x, 0.5 * inv_sqrt, -0.25 * inv_sqrt / x.value());
    return result;
}

inline Variable pow(const Variable& x, const double a)
{
    HyperGraph* graph = x.graph();

    const double pow = std::pow(x.value(), a);
    const Variable result = graph->new_tmp_variable(pow);
    graph->add_edge(result, x, a * std::pow(x.value(), a - 1.0), a * (a - 1.0) * std::pow(x.value(), a - 2.0));
    return result;
}

inline Variable exp(const Variable& x)
{
    HyperGraph* graph = x.graph();

    const double exp = std::exp(x.value());
    const Variable result = graph->new_tmp_variable(exp);
    graph->add_edge(result, x, exp, exp);
    return result;
}

inline Variable log(const Variable& x)
{
    HyperGraph* graph = x.graph();

    const double log = std::log(x.value());
    const Variable result = graph->new_tmp_variable(log);
    const double inv = 1.0 / x.value();
    graph->add_edge(result, x, inv, -inv * inv);
    return result;
}

inline Variable cos(const Variable& x)
{
    HyperGraph* graph = x.graph();

    const double cos = std::cos(x.value());
    const Variable result = graph->new_tmp_variable(cos);
    graph->add_edge(result, x, -std::sin(x.value()), -cos);
    return result;
}

inline Variable sin(const Variable& x)
{
    HyperGraph* graph = x.graph();

    const double sin = std::sin(x.value());
    const Variable result = graph->new_tmp_variable(sin);
    graph->add_edge(result, x, std::cos(x.value()), -sin);
    return result;
}

inline Variable tan(const Variable& x)
{
    HyperGraph* graph = x.graph();

    const double tan = std::tan(x.value());
    const double sec = 1.0 / std::cos(x.value());
    const double sec_sq = sec * sec;
    const Variable result = graph->new_tmp_variable(tan);
    graph->add_edge(result, x, sec_sq, 2.0 * tan * sec_sq);
    return result;
}

inline Variable acos(const Variable& x)
{
    HyperGraph* graph = x.graph();

    const double acos = std::acos(x.value());
    const Variable result = graph->new_tmp_variable(acos);
    const double tmp = 1.0 / (1.0 - x.value() * x.value());
    const double neg_tmp_sqrt = -std::sqrt(tmp);
    graph->add_edge(result, x, neg_tmp_sqrt, x.value() * neg_tmp_sqrt * tmp);
    return result;
}

inline Variable asin(const Variable& x)
{
    HyperGraph* graph = x.graph();

    const double asin = std::asin(x.value());
    const Variable result = graph->new_tmp_variable(asin);
    const double tmp = 1.0 / (1.0 - x.value() * x.value());
    const double tmp_sqrt = std::sqrt(tmp);
    graph->add_edge(result, x, tmp_sqrt, x.value() * tmp_sqrt * tmp);
    return result;
}

inline Variable atan(const Variable& x)
{
    HyperGraph* graph = x.graph();

    const double atan = std::atan(x.value());
    const Variable result = graph->new_tmp_variable(atan);
    const double tmp = 1 / (1 + x.value() * x.value());
    graph->add_edge(result, x, tmp, -2 * x.value() * tmp * tmp);
    return result;
}

} // namespace hypergraph
