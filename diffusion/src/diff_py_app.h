// SPDX-FileCopyrightText: 2025 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "mpicpp-lite/mpicpp-lite.h"
#include "opensn/python/lib/py_app.h"

namespace mpi = mpicpp_lite;

namespace diffpy
{

class DiffApp : public opensnpy::PyApp
{
public:
  explicit DiffApp(const mpi::Communicator& comm);
};

} // namespace diffpy
