// SPDX-FileCopyrightText: 2025 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "rom_py_app.h"
#include "py_wrappers.h"
#include "python/lib/console.h"

using namespace opensn;

namespace rompy
{

ROMApp::ROMApp(const mpi::Communicator& comm)
  : opensnpy::PyApp(comm)
{
  opensnpy::console.BindModule(WrapROM);
}

} // namespace rompy