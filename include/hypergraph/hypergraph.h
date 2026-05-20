//     __  __                      ______                 __
//    / / / /_  ______  ___  _____/ ____/________ _____  / /_
//   / /_/ / / / / __ \/ _ \/ ___/ / __/ ___/ __ `/ __ \/ __ \
//  / __  / /_/ / /_/ /  __/ /  / /_/ / /  / /_/ / /_/ / / / /
// /_/ /_/\__, / .___/\___/_/   \____/_/   \__,_/ .___/_/ /_/
//       /____/_/                              /_/
//
// Copyright (c) 2019 Thomas Oberbichler

#pragma once

#define _USE_MATH_DEFINES
#include <cmath>

#include <Eigen/Core>

#include <tsl/robin_map.h>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <tuple>

namespace hypergraph {

#if defined(_MSC_VER)
#define HYPERGRAPH_INLINE __forceinline
#else
#define HYPERGRAPH_INLINE __attribute__((always_inline)) inline
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

template <typename T>
HYPERGRAPH_INLINE Variable<T> exp(const Variable<T>& x);

template <typename T>
HYPERGRAPH_INLINE Variable<T> log(const Variable<T>& x);

template <typename T>
HYPERGRAPH_INLINE Variable<T> square(const Variable<T>& x);

template <typename T>
HYPERGRAPH_INLINE Variable<T> sinh(const Variable<T>& x);

template <typename T>
HYPERGRAPH_INLINE Variable<T> cosh(const Variable<T>& x);

template <typename T>
HYPERGRAPH_INLINE Variable<T> tanh(const Variable<T>& x);

template <typename T>
HYPERGRAPH_INLINE Variable<T> asinh(const Variable<T>& x);

template <typename T>
HYPERGRAPH_INLINE Variable<T> acosh(const Variable<T>& x);

template <typename T>
HYPERGRAPH_INLINE Variable<T> atanh(const Variable<T>& x);

template <typename T>
HYPERGRAPH_INLINE Variable<T> atan2(const Variable<T>& y, const Variable<T>& x);

template <typename T>
HYPERGRAPH_INLINE Variable<T> log2(const Variable<T>& x);

template <typename T>
HYPERGRAPH_INLINE Variable<T> log10(const Variable<T>& x);

template <typename T>
HYPERGRAPH_INLINE Variable<T> erf(const Variable<T>& x);

template <typename T>
HYPERGRAPH_INLINE Variable<T> erfc(const Variable<T>& x);

template <typename T>
HYPERGRAPH_INLINE Variable<T> sigmoid(const Variable<T>& x);

template <typename T>
HYPERGRAPH_INLINE Variable<T> softplus(const Variable<T>& x);

template <typename T>
HYPERGRAPH_INLINE Variable<T> min(const Variable<T>& x, const Variable<T>& y);

template <typename T>
HYPERGRAPH_INLINE Variable<T> max(const Variable<T>& x, const Variable<T>& y);

template <typename T>
HYPERGRAPH_INLINE Variable<T> pow(const Variable<T>& x, const Variable<T>& y);

template <typename T>
HYPERGRAPH_INLINE Variable<T> hypot(const Variable<T>& x, const Variable<T>& y);

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
    HyperGraph<T>* m_graph = nullptr;
    index m_id = 0;
    T m_value = T{};

public: // constructor
    Variable() = default;

    Variable(HyperGraph<T>* graph, const T value, const index id)
        : m_graph(graph)
        , m_id(id)
        , m_value(value)
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
        using namespace nanobind::literals;
        namespace nb = nanobind;

        const std::string name = "Variable";

        nb::class_<Type>(m, name.c_str())
            // properties
            .def_prop_ro("value", &Type::value)
            // read-only properties
            .def_prop_ro("_id", &Type::id)
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
            .def("exp", [](const Type& x) { return hypergraph::exp(x); })
            .def("log", [](const Type& x) { return hypergraph::log(x); })
            .def("square", [](const Type& x) { return hypergraph::square(x); })
            .def("sinh", [](const Type& x) { return hypergraph::sinh(x); })
            .def("cosh", [](const Type& x) { return hypergraph::cosh(x); })
            .def("tanh", [](const Type& x) { return hypergraph::tanh(x); })
            .def("arcsinh", [](const Type& x) { return hypergraph::asinh(x); })
            .def("arccosh", [](const Type& x) { return hypergraph::acosh(x); })
            .def("arctanh", [](const Type& x) { return hypergraph::atanh(x); })
            .def("log2", [](const Type& x) { return hypergraph::log2(x); })
            .def("log10", [](const Type& x) { return hypergraph::log10(x); })
            .def("erf", [](const Type& x) { return hypergraph::erf(x); })
            .def("erfc", [](const Type& x) { return hypergraph::erfc(x); })
            .def("sigmoid", [](const Type& x) { return hypergraph::sigmoid(x); })
            .def("softplus", [](const Type& x) { return hypergraph::softplus(x); })
            .def("__pow__", [](const Type& x, const Type& y) { return hypergraph::pow(x, y); })
            // operators
            .def(nb::self == nb::self)
            .def(nb::self != nb::self)
            .def(nb::self < nb::self)
            .def(nb::self > nb::self)
            .def(nb::self <= nb::self)
            .def(nb::self >= nb::self)
            .def(nb::self == double())
            .def(nb::self != double())
            .def(nb::self < double())
            .def(nb::self > double())
            .def(nb::self <= double())
            .def(nb::self >= double())
            .def(double() == nb::self)
            .def(double() != nb::self)
            .def(double() < nb::self)
            .def(double() > nb::self)
            .def(double() <= nb::self)
            .def(double() >= nb::self)
            .def(-nb::self)
            .def(nb::self + nb::self)
            .def(nb::self + double())
            .def(double() + nb::self)
            .def(nb::self += nb::self)
            .def(nb::self - nb::self)
            .def(nb::self - double())
            .def(double() - nb::self)
            .def(nb::self -= nb::self)
            .def(nb::self * nb::self)
            .def(nb::self * double())
            .def(double() * nb::self)
            .def(nb::self *= nb::self)
            .def(nb::self / nb::self)
            .def(nb::self / double())
            .def(double() / nb::self)
            .def(nb::self /= nb::self);
    }
};

template <typename T>
class Edge {
private: // types
    using Type = Edge<T>;

private: // variables
    index m_to = 0;
    T m_weight = T{};

public: // constructors
    Edge() = default;

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

    index num_variables() const
    {
        return length(m_variables);
    }

    std::tuple<std::vector<index>, std::vector<index>, std::vector<T>>
    h_sparse_triplets(const bool full = false) const
    {
        const index n = length(m_variables);

        std::vector<index> rows;
        std::vector<index> cols;
        std::vector<T> values;

        // Build a map from vertex_id → variable_index for efficient lookup
        tsl::robin_map<index, index> vid_to_var;
        vid_to_var.reserve(n);
        for (index i = 0; i < n; i++) {
            vid_to_var[m_variables[i].id()] = i;
        }

        // Diagonal entries from m_self_second_order_edges
        for (index i = 0; i < n; i++) {
            const T val = m_self_second_order_edges[m_variables[i].id()];
            if (val != T{}) {
                rows.push_back(i);
                cols.push_back(i);
                values.push_back(double(val));
            }
        }

        // Off-diagonal entries from m_second_order_edges
        // The structure stores entries with key = min(vid_a, vid_b) inside
        // m_second_order_edges[max(vid_a, vid_b)].
        for (index outer_vid = 0; outer_vid < length(m_second_order_edges); outer_vid++) {
            const auto& btree = m_second_order_edges[outer_vid];
            if (btree.empty()) {
                continue;
            }

            // Check if the outer vertex_id corresponds to a variable
            auto outer_it = vid_to_var.find(outer_vid);
            if (outer_it == vid_to_var.end()) {
                continue;
            }
            const index outer_var_idx = outer_it->second;

            for (const auto& [inner_vid, val] : btree) {
                if (val == T{}) {
                    continue;
                }

                auto inner_it = vid_to_var.find(inner_vid);
                if (inner_it == vid_to_var.end()) {
                    continue;
                }
                const index inner_var_idx = inner_it->second;

                // inner_vid < outer_vid by construction (minmax in second_order_edge)
                // So inner_var_idx may or may not be < outer_var_idx
                const auto [row, col] = hypergraph::minmax(inner_var_idx, outer_var_idx);

                // Upper triangle: row <= col
                rows.push_back(row);
                cols.push_back(col);
                values.push_back(double(val));

                if (full && row != col) {
                    rows.push_back(col);
                    cols.push_back(row);
                    values.push_back(double(val));
                }
            }
        }

        return {std::move(rows), std::move(cols), std::move(values)};
    }

public: // python
    template <typename TModule>
    static void register_python(TModule& m)
    {
        using namespace nanobind::literals;
        namespace nb = nanobind;

        const std::string name = "HyperGraph";

        nb::class_<Type>(m, name.c_str())
            // constructors
            .def(nb::init<>())
            // properties
            .def_prop_ro("num_variables", &Type::num_variables)
            // methods
            .def("new_variable", &Type::new_variable, "value"_a)
            .def("new_variables", &Type::new_variables, "values"_a)
            .def("compute", &Type::compute, "expression"_a)
            .def("g", nb::overload_cast<>(&Type::g, nb::const_))
            .def("g", nb::overload_cast<Eigen::Ref<Eigen::VectorXd>>(&Type::g, nb::const_), "out"_a)
            .def("h", nb::overload_cast<const bool>(&Type::h, nb::const_), "full"_a = false)
            .def("h", nb::overload_cast<Eigen::Ref<Eigen::MatrixXd>, const bool>(&Type::h, nb::const_), "out"_a, "full"_a = false)
            .def("h_sparse_triplets", &Type::h_sparse_triplets, "full"_a = false)
            .def("h_sparse", [](const Type& self, const std::string& format, const bool full) -> nb::object {
                auto [rows, cols, values] = self.h_sparse_triplets(full);
                const auto n = self.num_variables();

                nb::module_ scipy_sparse = nb::module_::import_("scipy.sparse");

                // Build numpy arrays from the triplet vectors
                nb::module_ np = nb::module_::import_("numpy");
                nb::object np_array = np.attr("array");

                nb::object py_rows = np_array(nb::cast(rows), "dtype"_a = np.attr("int64"));
                nb::object py_cols = np_array(nb::cast(cols), "dtype"_a = np.attr("int64"));
                nb::object py_vals = np_array(nb::cast(values), "dtype"_a = np.attr("float64"));

                nb::tuple data_ij = nb::make_tuple(py_vals, nb::make_tuple(py_rows, py_cols));
                nb::tuple shape = nb::make_tuple(n, n);

                nb::object coo = scipy_sparse.attr("coo_matrix")(data_ij, "shape"_a = shape);

                if (format == "coo") {
                    return coo;
                } else if (format == "csc") {
                    return coo.attr("tocsc")();
                } else if (format == "csr") {
                    return coo.attr("tocsr")();
                } else {
                    throw std::invalid_argument("h_sparse: format must be 'coo', 'csc', or 'csr'");
                }
            }, "format"_a = "csc", "full"_a = false);
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
#ifdef HYPERGRAPH_EXCEPTIONS
    if (x.value() == 0.0) {
        throw std::domain_error("inv: division by zero");
    }
#endif

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
using std::acosh;
using std::asin;
using std::asinh;
using std::atan;
using std::atan2;
using std::atanh;
using std::cos;
using std::cosh;
using std::erf;
using std::erfc;
using std::exp;
using std::hypot;
using std::log;
using std::log2;
using std::log10;
using std::pow;
using std::sin;
using std::sinh;
using std::sqrt;
using std::tan;
using std::tanh;

template <typename T>
HYPERGRAPH_INLINE Variable<T> abs(const Variable<T>& x)
{
    if (x.value() > 0) {
        return x;
    } else if (x.value() < 0) {
        return -x;
    } else {
        // Subgradient convention: abs(0) = 0, d|x|/dx = 0 at x = 0
        HyperGraph<T>* graph = x.graph();
        const Variable<T> result = graph->new_tmp_variable(0.0);
        graph->add_edge(result, x, 0.0, 0.0);
        return result;
    }
}

template <typename T>
HYPERGRAPH_INLINE Variable<T> square(const Variable<T>& x)
{
    HyperGraph<T>* graph = x.graph();

    const auto sq = x.value() * x.value();
    const Variable<T> result = graph->new_tmp_variable(sq);
    graph->add_edge(result, x, 2.0 * x.value(), 2.0);
    return result;
}

template <typename T>
HYPERGRAPH_INLINE Variable<T> sqrt(const Variable<T>& x)
{
    using std::sqrt;

#ifdef HYPERGRAPH_EXCEPTIONS
    if (x.value() < 0.0) {
        throw std::domain_error("sqrt: negative argument");
    }
    if (x.value() == 0.0) {
        throw std::domain_error("sqrt: derivative undefined at x = 0");
    }
#endif

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

#ifdef HYPERGRAPH_EXCEPTIONS
    if (x.value() == 0.0 && a < 2.0) {
        throw std::domain_error("pow: derivative undefined at x = 0 for exponent < 2");
    }
#endif

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

#ifdef HYPERGRAPH_EXCEPTIONS
    if (x.value() <= 0.0) {
        throw std::domain_error("log: argument must be positive");
    }
#endif

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
    using std::cos;
    using std::tan;

#ifdef HYPERGRAPH_EXCEPTIONS
    const auto cos_check = cos(x.value());
    if (cos_check == 0.0) {
        throw std::domain_error("tan: derivative undefined at x = pi/2 + n*pi");
    }
#endif

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

#ifdef HYPERGRAPH_EXCEPTIONS
    if (x.value() <= -1.0 || x.value() >= 1.0) {
        throw std::domain_error("acos: derivative undefined at |x| >= 1");
    }
#endif

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

#ifdef HYPERGRAPH_EXCEPTIONS
    if (x.value() <= -1.0 || x.value() >= 1.0) {
        throw std::domain_error("asin: derivative undefined at |x| >= 1");
    }
#endif

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

template <typename T>
HYPERGRAPH_INLINE Variable<T> sinh(const Variable<T>& x)
{
    using std::cosh;
    using std::sinh;

    HyperGraph<T>* graph = x.graph();

    const auto sinh_x = sinh(x.value());
    const Variable<T> result = graph->new_tmp_variable(sinh_x);
    graph->add_edge(result, x, cosh(x.value()), sinh_x);
    return result;
}

template <typename T>
HYPERGRAPH_INLINE Variable<T> cosh(const Variable<T>& x)
{
    using std::cosh;
    using std::sinh;

    HyperGraph<T>* graph = x.graph();

    const auto cosh_x = cosh(x.value());
    const Variable<T> result = graph->new_tmp_variable(cosh_x);
    graph->add_edge(result, x, sinh(x.value()), cosh_x);
    return result;
}

template <typename T>
HYPERGRAPH_INLINE Variable<T> tanh(const Variable<T>& x)
{
    using std::tanh;

    HyperGraph<T>* graph = x.graph();

    const auto tanh_x = tanh(x.value());
    const auto sech_sq = 1.0 - tanh_x * tanh_x;
    const Variable<T> result = graph->new_tmp_variable(tanh_x);
    graph->add_edge(result, x, sech_sq, -2.0 * tanh_x * sech_sq);
    return result;
}

template <typename T>
HYPERGRAPH_INLINE Variable<T> asinh(const Variable<T>& x)
{
    using std::asinh;
    using std::sqrt;

    HyperGraph<T>* graph = x.graph();

    const auto asinh_x = asinh(x.value());
    const auto tmp = 1.0 / (1.0 + x.value() * x.value());
    const auto tmp_sqrt = sqrt(tmp);
    const Variable<T> result = graph->new_tmp_variable(asinh_x);
    graph->add_edge(result, x, tmp_sqrt, -x.value() * tmp_sqrt * tmp);
    return result;
}

template <typename T>
HYPERGRAPH_INLINE Variable<T> acosh(const Variable<T>& x)
{
    using std::acosh;
    using std::sqrt;

#ifdef HYPERGRAPH_EXCEPTIONS
    if (x.value() <= 1.0) {
        throw std::domain_error("acosh: argument must be > 1");
    }
#endif

    HyperGraph<T>* graph = x.graph();

    const auto acosh_x = acosh(x.value());
    const auto tmp = 1.0 / (x.value() * x.value() - 1.0);
    const auto tmp_sqrt = sqrt(tmp);
    const Variable<T> result = graph->new_tmp_variable(acosh_x);
    graph->add_edge(result, x, tmp_sqrt, -x.value() * tmp_sqrt * tmp);
    return result;
}

template <typename T>
HYPERGRAPH_INLINE Variable<T> atanh(const Variable<T>& x)
{
    using std::atanh;

#ifdef HYPERGRAPH_EXCEPTIONS
    if (x.value() <= -1.0 || x.value() >= 1.0) {
        throw std::domain_error("atanh: argument must be in (-1, 1)");
    }
#endif

    HyperGraph<T>* graph = x.graph();

    const auto atanh_x = atanh(x.value());
    const auto one_minus_x2 = 1.0 - x.value() * x.value();
    const auto tmp = 1.0 / one_minus_x2;
    const Variable<T> result = graph->new_tmp_variable(atanh_x);
    graph->add_edge(result, x, tmp, 2.0 * x.value() * tmp * tmp);
    return result;
}

template <typename T>
HYPERGRAPH_INLINE Variable<T> atan2(const Variable<T>& y, const Variable<T>& x)
{
    using std::atan2;

#ifdef HYPERGRAPH_EXCEPTIONS
    if (x.value() == 0.0 && y.value() == 0.0) {
        throw std::domain_error("atan2: undefined at (0, 0)");
    }
    if (x.value() == 0.0) {
        throw std::domain_error("atan2: derivative undefined at x = 0");
    }
#endif

    // Use composition atan(y/x) for correct automatic second derivatives.
    // The derivatives of atan2(y,x) and atan(y/x) are identical for x != 0.
    Variable<T> result = hypergraph::atan(y / x);

    // Correct the value to match atan2 semantics (handles all quadrants)
    result.set_value(atan2(y.value(), x.value()));

    return result;
}

template <typename T>
HYPERGRAPH_INLINE Variable<T> log2(const Variable<T>& x)
{
    using std::log;
    using std::log2;

#ifdef HYPERGRAPH_EXCEPTIONS
    if (x.value() <= 0.0) {
        throw std::domain_error("log2: argument must be positive");
    }
#endif

    HyperGraph<T>* graph = x.graph();

    const auto log2_x = log2(x.value());
    const auto inv = 1.0 / x.value();
    const auto ln2 = log(2.0);
    const auto inv_ln2 = 1.0 / ln2;
    const Variable<T> result = graph->new_tmp_variable(log2_x);
    // d(log2(x))/dx = 1/(x·ln2), d²(log2(x))/dx² = -1/(x²·ln2)
    graph->add_edge(result, x, inv * inv_ln2, -inv * inv * inv_ln2);
    return result;
}

template <typename T>
HYPERGRAPH_INLINE Variable<T> log10(const Variable<T>& x)
{
    using std::log;
    using std::log10;

#ifdef HYPERGRAPH_EXCEPTIONS
    if (x.value() <= 0.0) {
        throw std::domain_error("log10: argument must be positive");
    }
#endif

    HyperGraph<T>* graph = x.graph();

    const auto log10_x = log10(x.value());
    const auto inv = 1.0 / x.value();
    const auto ln10 = log(10.0);
    const auto inv_ln10 = 1.0 / ln10;
    const Variable<T> result = graph->new_tmp_variable(log10_x);
    // d(log10(x))/dx = 1/(x·ln10), d²(log10(x))/dx² = -1/(x²·ln10)
    graph->add_edge(result, x, inv * inv_ln10, -inv * inv * inv_ln10);
    return result;
}

template <typename T>
HYPERGRAPH_INLINE Variable<T> erf(const Variable<T>& x)
{
    using std::erf;
    using std::exp;

    HyperGraph<T>* graph = x.graph();

    const auto erf_x = erf(x.value());
    // d(erf(x))/dx = 2/sqrt(π) · exp(-x²)
    // d²(erf(x))/dx² = -4x/sqrt(π) · exp(-x²)
    const auto two_over_sqrt_pi = 2.0 / std::sqrt(M_PI);
    const auto exp_neg_x2 = exp(-x.value() * x.value());
    const auto first_deriv = two_over_sqrt_pi * exp_neg_x2;
    const auto second_deriv = -2.0 * x.value() * first_deriv;
    const Variable<T> result = graph->new_tmp_variable(erf_x);
    graph->add_edge(result, x, first_deriv, second_deriv);
    return result;
}

template <typename T>
HYPERGRAPH_INLINE Variable<T> erfc(const Variable<T>& x)
{
    using std::erfc;
    using std::exp;

    HyperGraph<T>* graph = x.graph();

    const auto erfc_x = erfc(x.value());
    // erfc(x) = 1 - erf(x), so derivatives are negated
    // d(erfc(x))/dx = -2/sqrt(π) · exp(-x²)
    // d²(erfc(x))/dx² = 4x/sqrt(π) · exp(-x²)
    const auto two_over_sqrt_pi = 2.0 / std::sqrt(M_PI);
    const auto exp_neg_x2 = exp(-x.value() * x.value());
    const auto first_deriv = -two_over_sqrt_pi * exp_neg_x2;
    const auto second_deriv = 2.0 * x.value() * two_over_sqrt_pi * exp_neg_x2;
    const Variable<T> result = graph->new_tmp_variable(erfc_x);
    graph->add_edge(result, x, first_deriv, second_deriv);
    return result;
}

template <typename T>
HYPERGRAPH_INLINE Variable<T> sigmoid(const Variable<T>& x)
{
    using std::exp;

    HyperGraph<T>* graph = x.graph();

    // sigmoid(x) = 1/(1+exp(-x))
    const auto exp_neg_x = exp(-x.value());
    const auto sig = 1.0 / (1.0 + exp_neg_x);
    // d(sigmoid)/dx = sigmoid · (1 - sigmoid)
    const auto first_deriv = sig * (1.0 - sig);
    // d²(sigmoid)/dx² = sigmoid · (1 - sigmoid) · (1 - 2·sigmoid)
    const auto second_deriv = first_deriv * (1.0 - 2.0 * sig);
    const Variable<T> result = graph->new_tmp_variable(sig);
    graph->add_edge(result, x, first_deriv, second_deriv);
    return result;
}

template <typename T>
HYPERGRAPH_INLINE Variable<T> softplus(const Variable<T>& x)
{
    using std::exp;
    using std::log;

    HyperGraph<T>* graph = x.graph();

    // softplus(x) = log(1 + exp(x))
    // For numerical stability: if x >> 0, softplus(x) ≈ x
    const auto exp_x = exp(x.value());
    const auto softplus_x = log(1.0 + exp_x);
    // d(softplus)/dx = sigmoid(x) = exp(x)/(1+exp(x)) = 1/(1+exp(-x))
    const auto sig = exp_x / (1.0 + exp_x);
    // d²(softplus)/dx² = sigmoid(x)·(1-sigmoid(x))
    const auto second_deriv = sig * (1.0 - sig);
    const Variable<T> result = graph->new_tmp_variable(softplus_x);
    graph->add_edge(result, x, sig, second_deriv);
    return result;
}

template <typename T>
HYPERGRAPH_INLINE Variable<T> min(const Variable<T>& x, const Variable<T>& y)
{
    // Subgradient convention: at x == y, derivative w.r.t. x is 1, w.r.t. y is 0
    if (x.value() <= y.value()) {
        HyperGraph<T>* graph = x.graph();
        const Variable<T> result = graph->new_tmp_variable(x.value());
        graph->add_edge(result, x, y, 1.0, 0.0, 0.0);
        return result;
    } else {
        HyperGraph<T>* graph = y.graph();
        const Variable<T> result = graph->new_tmp_variable(y.value());
        graph->add_edge(result, x, y, 0.0, 1.0, 0.0);
        return result;
    }
}

template <typename T>
HYPERGRAPH_INLINE Variable<T> max(const Variable<T>& x, const Variable<T>& y)
{
    // Subgradient convention: at x == y, derivative w.r.t. x is 1, w.r.t. y is 0
    if (x.value() >= y.value()) {
        HyperGraph<T>* graph = x.graph();
        const Variable<T> result = graph->new_tmp_variable(x.value());
        graph->add_edge(result, x, y, 1.0, 0.0, 0.0);
        return result;
    } else {
        HyperGraph<T>* graph = y.graph();
        const Variable<T> result = graph->new_tmp_variable(y.value());
        graph->add_edge(result, x, y, 0.0, 1.0, 0.0);
        return result;
    }
}

template <typename T>
HYPERGRAPH_INLINE Variable<T> pow(const Variable<T>& x, const Variable<T>& y)
{
    using std::pow;

#ifdef HYPERGRAPH_EXCEPTIONS
    if (x.value() <= 0.0) {
        throw std::domain_error("pow(Variable, Variable): base must be positive");
    }
#endif

    // x^y = exp(y·ln(x)) — compose from existing primitives for correct
    // second-order derivatives (self and cross terms)
    Variable<T> result = hypergraph::exp(y * hypergraph::log(x));

    // Correct the value to use std::pow for better numerical accuracy
    result.set_value(pow(x.value(), y.value()));

    return result;
}

template <typename T>
HYPERGRAPH_INLINE Variable<T> hypot(const Variable<T>& x, const Variable<T>& y)
{
    using std::hypot;

#ifdef HYPERGRAPH_EXCEPTIONS
    if (x.value() == 0.0 && y.value() == 0.0) {
        throw std::domain_error("hypot: derivative undefined at (0, 0)");
    }
#endif

    // sqrt(x² + y²) — compose from existing primitives for correct
    // second-order derivatives (self and cross terms)
    Variable<T> result = hypergraph::sqrt(hypergraph::square(x) + hypergraph::square(y));

    // Correct the value to use std::hypot for better numerical accuracy
    result.set_value(hypot(x.value(), y.value()));

    return result;
}

} // namespace hypergraph