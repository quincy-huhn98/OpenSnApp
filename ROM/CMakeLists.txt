cmake_minimum_required(VERSION 3.14)
project(OpenSnApp LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

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

find_package(Boost CONFIG)
if (NOT Boost_FOUND)
  message(STATUS "Boost CONFIG not found, falling back to module mode")
  find_package(Boost REQUIRED)
endif()

find_library(LIBROM_LIBRARY
  NAMES ROM
  HINTS
    ENV LIBROM_DIR
    ${LIBROM_DIR}
    ${LIBROM_DIR}/install
  PATH_SUFFIXES lib
  REQUIRED)

find_path(LIBROM_INCLUDE_DIR
  NAMES librom.h
  HINTS
    ENV LIBROM_DIR
    ${LIBROM_DIR}
    ${LIBROM_DIR}/install
  PATH_SUFFIXES include
  REQUIRED)

message(STATUS "libROM include dir: ${LIBROM_INCLUDE_DIR}")
message(STATUS "libROM library: ${LIBROM_LIBRARY}")

file(GLOB_RECURSE ROM_SRCS CONFIGURE_DEPENDS ${PROJECT_SOURCE_DIR}/src/rom/*.cc)

# Python module
add_library(romapp MODULE
    src/main.cc
    src/rom_py_app.cc
    src/rom.cc
    ${ROM_SRCS}
)

execute_process(
    COMMAND python3 -m pybind11 --includes
    OUTPUT_VARIABLE PYBIND11_INCLUDE_FLAGS
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
string(REPLACE "-I" "" PYBIND11_INCLUDE_DIRS "${PYBIND11_INCLUDE_FLAGS}")
separate_arguments(PYBIND11_INCLUDE_DIRS)

target_include_directories(romapp 
    PRIVATE 
    ${PROJECT_SOURCE_DIR}/rom
    ${PYBIND11_INCLUDE_DIRS}
    ${OpenSn_INSTALL_PREFIX}/include/opensn
    ${LIBROM_INCLUDE_DIR}
    ${Boost_INCLUDE_DIRS}
)
target_link_libraries(romapp 
    PRIVATE 
    opensn::opensn
    opensn::libopensnpy
    MPI::MPI_CXX 
    caliper 
    pybind11::module 
    Python::Python 
    ${LIBROM_LIBRARY})

set_target_properties(romapp PROPERTIES
    PREFIX ""
    OUTPUT_NAME "romapp"
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python
)

add_executable(rom_app_exec src/main.cc src/rom_py_app.cc src/rom.cc ${ROM_SRCS})
target_include_directories(rom_app_exec PRIVATE
    ${PROJECT_SOURCE_DIR}/rom
    ${PYBIND11_INCLUDE_DIRS}
    ${OpenSn_INSTALL_PREFIX}/include/opensn
    ${LIBROM_INCLUDE_DIR}
    ${Boost_INCLUDE_DIRS}
)
target_link_libraries(rom_app_exec 
    PRIVATE 
    opensn::opensn
    opensn::libopensnpy
    MPI::MPI_C 
    caliper 
    pybind11::module 
    Python::Python 
    ${LIBROM_LIBRARY})
