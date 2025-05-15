#include "test.h"
#include "opensn/python/lib/py_app.h"
#include "opensn/mpicpp-lite/mpicpp-lite.h"

int main(int argc, char** argv)
{
    mpi::Environment env(argc, argv);
    py::scoped_interpreter guard{};
    auto app = TestApp::Create({});
    return 0;
}
