// SPDX-FileCopyrightText: 2025 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "linalg/Vector.h"

namespace opensn
{

enum class Phase
{
  OFFLINE,
  MERGE,
  SYSTEMS,
  ONLINE
};

struct ROMOptions
{
  int param_id = 0;
  Phase phase = Phase::OFFLINE;
  std::string param_file = "";
  std::unique_ptr<CAROM::Vector> new_point;

  ROMOptions() = default;
};

} // namespace opensn