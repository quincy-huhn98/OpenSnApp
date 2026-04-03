// SPDX-FileCopyrightText: 2025 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "opensn/framework/parameters/parameter_block.h"
#include "opensn/framework/data_types/vector.h"
#include <pybind11/pybind11.h>
#include <vector>

namespace py = pybind11;
namespace opensn
{

// Wrap the diffusion components of OpenSn
void py_diffusion(py::module& pyopensn);
void WrapDiffusion(py::module& diffusion);

/// Translate a Python dictionary into a ParameterBlock.
ParameterBlock kwargs_to_param_block(const py::kwargs& params);

} // namespace opensn
