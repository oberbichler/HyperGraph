#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <hypergraph/hypergraph.h>
#include "hypergraph/counting.h"

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

    hypergraph::Counting::register_python(m);
}
