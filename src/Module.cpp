#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <hypergraph/hypergraph.h>

PYBIND11_MODULE(hypergraph, m) {
    m.doc() = "HyperJet by Thomas Oberbichler";
    m.attr("__author__") = "Thomas Oberbichler";
    m.attr("__copyright__") = "Copyright (c) 2019, Thomas Oberbichler";
    m.attr("__version__") = HYPERGRAPH_VERSION;
    m.attr("__email__") = "thomas.oberbichler@gmail.com";
    m.attr("__status__") = "Development";

    namespace py = pybind11;
    using namespace pybind11::literals;

    hypergraph::HyperGraph<double>::register_python(m);
    hypergraph::Variable<double>::register_python(m);

    using Var = hypergraph::Variable<double>;
    m.def("atan2", [](const Var& y, const Var& x) { return hypergraph::atan2(y, x); }, "y"_a, "x"_a);
    m.def("log2", [](const Var& x) { return hypergraph::log2(x); }, "x"_a);
    m.def("log10", [](const Var& x) { return hypergraph::log10(x); }, "x"_a);
    m.def("erf", [](const Var& x) { return hypergraph::erf(x); }, "x"_a);
    m.def("erfc", [](const Var& x) { return hypergraph::erfc(x); }, "x"_a);
    m.def("sigmoid", [](const Var& x) { return hypergraph::sigmoid(x); }, "x"_a);
    m.def("softplus", [](const Var& x) { return hypergraph::softplus(x); }, "x"_a);
    m.def("min", [](const Var& x, const Var& y) { return hypergraph::min(x, y); }, "x"_a, "y"_a);
    m.def("max", [](const Var& x, const Var& y) { return hypergraph::max(x, y); }, "x"_a, "y"_a);
    m.def("pow", [](const Var& x, const Var& y) { return hypergraph::pow(x, y); }, "x"_a, "y"_a);
    m.def("hypot", [](const Var& x, const Var& y) { return hypergraph::hypot(x, y); }, "x"_a, "y"_a);
}
