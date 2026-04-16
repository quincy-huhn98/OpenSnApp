// SPDX-FileCopyrightText: 2025 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "diff_py_app.h"
#include "py_wrappers.h"
#include "opensn/python/lib/console.h"

using namespace opensn;

namespace diffpy
{

DiffApp::DiffApp(const mpi::Communicator& comm)
  : opensnpy::PyApp(comm)
{
  opensnpy::console.BindModule(WrapDiffusion);
}

} // namespace diffpy
