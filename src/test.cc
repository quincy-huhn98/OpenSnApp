#include "test.h"
#include "opensn/framework/logging/log.h"
#include "opensn/framework/object_factory.h"

using namespace opensn;

OpenSnRegisterObject(TestApp);

InputParameters
TestApp::GetInputParameters()
{
  InputParameters params;
  params.SetGeneralDescription("A dummy application for testing.");
  params.AddOptionalParameter("name", "test_app", "A name for the application.");
  return params;
}

TestApp::TestApp(const InputParameters& params) : name_(params.GetParamValue<std::string>("name"))
{
  opensn::log.Log() << "TestApp named " << name_ << " created.";
}

std::shared_ptr<TestApp>
TestApp::Create(const ParameterBlock& params)
{
  auto& factory = opensn::ObjectFactory::GetInstance();
  return factory.Create<TestApp>("TestApp", params);
}
