// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "linalg/Vector.h"

namespace opensn
{

struct ROMOptions
{
  int param_id = 0;
  std::string phase = "offline";
  std::string param_file = "";
  std::unique_ptr<CAROM::Vector> new_point;

  ROMOptions() = default;
};

} // namespace opensn