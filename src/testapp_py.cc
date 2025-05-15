#include <pybind11/pybind11.h>
#include "test.h"

namespace py = pybind11;

PYBIND11_MODULE(myapp, m)
{
    m.doc() = "Python bindings for my OpenSn-based TestApp";

    py::class_<TestApp, std::shared_ptr<TestApp>>(m, "TestApp")
        .def_static("Create", &TestApp::Create);
}
