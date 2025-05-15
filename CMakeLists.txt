cmake_minimum_required(VERSION 3.14)
project(MyOpenSnApp LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(MPI REQUIRED)
find_package(VTK REQUIRED COMPONENTS
    CommonCore
    CommonDataModel
    IOLegacy
    IOCore
    IOXML
    ParallelCore
    IOParallelXML
    FiltersCore
    IOEnSight
    IOExodus
)
find_package(OpenSn REQUIRED)

# Optional: find pybind11
find_package(pybind11 REQUIRED)

find_package(caliper REQUIRED)

find_package(Python REQUIRED COMPONENTS Interpreter Development)

# Python module
add_library(testapp MODULE
    src/test.cc
    src/testapp_py.cc
)

execute_process(
    COMMAND python3 -m pybind11 --includes
    OUTPUT_VARIABLE PYBIND11_INCLUDE_FLAGS
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
string(REPLACE "-I" "" PYBIND11_INCLUDE_DIRS "${PYBIND11_INCLUDE_FLAGS}")
separate_arguments(PYBIND11_INCLUDE_DIRS)

target_include_directories(testapp PRIVATE ${OPENSN_INCLUDE_DIR} ${PYBIND11_INCLUDE_DIRS})
target_link_libraries(testapp PRIVATE opensn::libopensn MPI::MPI_C caliper pybind11::module Python::Python)

set_target_properties(testapp PROPERTIES
    PREFIX ""    # remove "lib" prefix
    OUTPUT_NAME "testapp"  # Python import name
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python
)

# Optional executable
add_executable(test_app_exec src/main.cc src/test.cc)
target_include_directories(test_app_exec PRIVATE ${OPENSN_INCLUDE_DIR} ${PYBIND11_INCLUDE_DIRS})
target_link_libraries(test_app_exec PRIVATE opensn::libopensn MPI::MPI_C caliper pybind11::module Python::Python)