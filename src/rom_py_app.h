// SPDX-FileCopyrightText: 2025 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "mpicpp-lite/mpicpp-lite.h"
#include "python/lib/py_app.h"

namespace mpi = mpicpp_lite;

namespace rompy
{

class ROMApp : public opensnpy::PyApp
{
public:
  explicit ROMApp(const mpi::Communicator& comm);
};

} // namespace rompy