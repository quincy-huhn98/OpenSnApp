cmake_minimum_required(VERSION 3.14)
project(OpenSnApp LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

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

include(CMakeFindDependencyMacro)

find_package(OpenSn REQUIRED)

message(STATUS "Found OpenSn ${OpenSn_VERSION} (from: ${OpenSn_DIR})")

find_package(pybind11 REQUIRED)

find_package(caliper REQUIRED)

find_package(Python REQUIRED COMPONENTS Interpreter Development)

file(GLOB_RECURSE DIFFUSION_SRCS CONFIGURE_DEPENDS ${PROJECT_SOURCE_DIR}/src/diffusion/*.cc)

add_executable(
  diff_app_exec
  src/main.cc
  src/diff_py_app.cc
  src/diffusion.cc
  ${DIFFUSION_SRCS}
)

target_include_directories(diff_app_exec PRIVATE
  ${PROJECT_SOURCE_DIR}/src
  ${OpenSn_INSTALL_PREFIX}/include
  ${PROJECT_SOURCE_DIR}/blah/opensn/external
)

target_link_libraries(diff_app_exec PRIVATE
  opensn::opensn
  opensn::libopensnpy
  MPI::MPI_CXX
  caliper
  pybind11::embed
  Python::Python
)
