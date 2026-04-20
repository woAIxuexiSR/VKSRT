#pragma once

#include "encoding.h"
#include <nlohmann/json.hpp>
#include <memory>
#include <string>

using json = nlohmann::json;

std::unique_ptr<Encoding> createEncoding(Device &device, const std::string &type,
                                         int inputDim, const json &params);
