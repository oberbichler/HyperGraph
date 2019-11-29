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
    std::vector<Vertex> m_vertices;
    std::vector<tsl::robin_map<index, double>> m_second_order_edges;
    std::vector<double> m_self_second_order_edges;

public: // constructor
public: // methods
    Variable new_variable(const double value)
    {
        const index id = length(m_vertices);
        m_vertices.push_back(Vertex(id));
        return Variable(this, value, id);
    }

    std::vector<Variable> new_variables(const std::vector<double>& values)
    {
        std::vector<Variable> variables(values.size());
        for (index i = 0; i < length(values); i++) {
            variables[i] = new_variable(values[i]);
        }
        return variables;
    }

    void add_edge(const Variable& c, const Variable& p, const double weight, const double second_order_weight)
    {
        Vertex& vertex = m_vertices[c.id()];
        vertex.set_edge1(Edge(p.id(), weight));
        vertex.set_second_order_weight(second_order_weight);
    }

    void add_edge(const Variable& c, const Variable& p1, const Variable& p2, const double weight1, const double weight2, const double second_order_weight)
    {
        Vertex& vertex = m_vertices[c.id()];
        vertex.set_edge1(Edge(p1.id(), weight1));
        vertex.set_edge2(Edge(p2.id(), weight2));
        vertex.set_second_order_weight(second_order_weight);
    }

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
        const auto [min, max] = std::minmax(vertex_id(a), vertex_id(b));
        return m_second_order_edges[max].at(min);
    }

    template <typename T>
    inline double& second_order_edge(T a, T b)
    {
        const auto [min, max] = std::minmax(vertex_id(a), vertex_id(b));
        return m_second_order_edges[max][min];
    }

    void set_adjoint(const Variable& v, const double adj)
    {
        m_vertices[v.id()].set_weight(adj);
    }

    double get_adjoint(const Variable& v)
    {
        return m_vertices[v.id()].weight();
    }

    double get_adjoint(const Variable& i, const Variable& j)
    {
        if (i.id() == j.id()) {
            return self_second_order_edge(i);
        } else {
            return second_order_edge(i, j);
        }
    }

    index single_edge_propagate(index x, double& a)
    {
        bool cont = m_vertices[x].edge1().to() != x && m_vertices[x].edge2().to() == x;

        while (cont) {
            a *= m_vertices[x].edge1().weight();
            x = m_vertices[x].edge1().to();
            cont = m_vertices[x].edge1().to() != x && m_vertices[x].edge2().to() == x;
        }

        return x;
    }

    void push_edge(const Edge& foEdge, const Edge& soEdge)
    {
        if (foEdge.to() == soEdge.to()) {
            self_second_order_edge(foEdge) += 2 * foEdge.weight() * soEdge.weight();
        } else {
            second_order_edge(foEdge, soEdge) = foEdge.weight() * soEdge.weight();
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
            Vertex& vertex = m_vertices[vid];
            Edge& e1 = vertex.edge1();
            Edge& e2 = vertex.edge2();

            if (e1.to() == vid) {
                continue;
            }

            auto& btree = m_second_order_edges[vid];

            if (e2.to() == vid) {
                for (const auto it : btree) {
                    Edge soEdge(it.first, it.second);
                    push_edge(e1, soEdge);
                }
            } else {
                for (const auto it : btree) {
                    Edge soEdge(it.first, it.second);
                    push_edge(e1, soEdge);
                    push_edge(e2, soEdge);
                }
            }
            if (self_second_order_edge(vid) != 0.0) {
                self_second_order_edge(e1) += e1.weight() * e1.weight() * self_second_order_edge(vid);
                if (e2.to() != vid) {
                    self_second_order_edge(e2) += e2.weight() * e2.weight() * self_second_order_edge(vid);
                    if (e1.to() == e2.to()) {
                        self_second_order_edge(e2) += 2.0 * e1.weight() * e2.weight() * self_second_order_edge(vid);
                    } else {
                        second_order_edge(e1, e2) = e1.weight() * e2.weight() * self_second_order_edge(vid);
                    }
                }
            }

            double a = vertex.weight();
            if (a != 0.0) {
                if (vertex.second_order_weight() != 0.0) {
                    if (e2.to() == vid) {
                        self_second_order_edge(e1) += a * vertex.second_order_weight();
                    } else if (e1.to() == e2.to()) {
                        self_second_order_edge(e1) += 2.0 * a * vertex.second_order_weight();
                    } else {
                        second_order_edge(e1, e2) = a * vertex.second_order_weight();
                    }
                }

                vertex.weight() = 0.0;
                m_vertices[e1.to()].weight() += a * e1.weight();
                if (e2.to() != vid) {
                    m_vertices[e2.to()].weight() += a * e2.weight();
                }
            }
        }
    }

    std::pair<Eigen::VectorXd, Eigen::MatrixXd> derive(const Variable expression, const std::vector<Variable>& variables)
    {
        set_adjoint(expression, 1.0);
        propagate_adjoint();

        const index n = length(variables);

        Eigen::VectorXd g(n);
        Eigen::MatrixXd h = Eigen::MatrixXd::Zero(n, n);

        for (index i = 0; i < n; i++) {
            g(i) = get_adjoint(variables[i]);
        }

        for (index i = 0; i < n; i++) {
            for (index j = i; j < n; j++) {
                h(i, j) = get_adjoint(variables[i], variables[j]);
            }
        }

        return {g, h};
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
            .def("derive", &Type::derive, "expression"_a, "variables"_a);
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

    Variable ret = graph->new_variable(-x.value());
    graph->add_edge(ret, x, -1.0, 0.0);
    return ret;
}

inline Variable operator+(const Variable& lhs, const Variable& rhs)
{
    HyperGraph* graph = lhs.graph();

    Variable result = graph->new_variable(lhs.value() + rhs.value());
    graph->add_edge(result, lhs, rhs, 1.0, 1.0, 0.0);
    return result;
}

inline Variable operator+(const Variable& lhs, const double rhs)
{
    HyperGraph* graph = lhs.graph();

    Variable result = graph->new_variable(lhs.value() + rhs);
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

    Variable ret = graph->new_variable(lhs.value() - rhs.value());
    graph->add_edge(ret, lhs, rhs, 1.0, -1.0, 0.0);
    return ret;
}

inline Variable operator-(const Variable& lhs, const double rhs)
{
    HyperGraph* graph = lhs.graph();

    Variable ret = graph->new_variable(lhs.value() - rhs);
    graph->add_edge(ret, lhs, 1.0, double(0.0));
    return ret;
}

inline Variable operator-(const double lhs, const Variable& rhs)
{
    HyperGraph* graph = rhs.graph();

    Variable ret = graph->new_variable(lhs - rhs.value());
    graph->add_edge(ret, rhs, double(-1.0), double(0.0));
    return ret;
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

    Variable ret = graph->new_variable(lhs.value() * rhs.value());
    graph->add_edge(ret, lhs, rhs, rhs.value(), lhs.value(), 1.0);
    return ret;
}

inline Variable operator*(const Variable& lhs, const double rhs)
{
    HyperGraph* graph = lhs.graph();

    Variable ret = graph->new_variable(lhs.value() * rhs);
    graph->add_edge(ret, lhs, rhs, double(0.0));
    return ret;
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

    double inv_x = 1.0 / x.value();
    double inv_x_sq = inv_x * inv_x;
    double inv_x_cu = inv_x_sq * inv_x;
    Variable ret = graph->new_variable(inv_x);
    graph->add_edge(ret, x, -inv_x_sq, 2.0 * inv_x_cu);
    return ret;
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

} // namespace hypergraph
