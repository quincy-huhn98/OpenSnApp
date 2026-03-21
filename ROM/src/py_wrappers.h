// SPDX-FileCopyrightText: 2025 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "framework/parameters/parameter_block.h"
#include <pybind11/pybind11.h>
#include <vector>

namespace py = pybind11;
namespace opensn
{

// Wrap the ROM components of app
void py_rom(py::module& pyopensn);
void WrapROM(py::module& ROM);

/// Translate a Python dictionary into a ParameterBlock.
ParameterBlock kwargs_to_param_block(const py::kwargs& params);

} // namespace opensn