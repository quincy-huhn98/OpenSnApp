#pragma once
#include "opensn/framework/object.h"

class TestApp : public opensn::Object
{
public:
  explicit TestApp(const opensn::InputParameters& params);

private:
  const std::string name_;

public:
  static opensn::InputParameters GetInputParameters();
  static std::shared_ptr<TestApp> Create(const opensn::ParameterBlock& params);
};
