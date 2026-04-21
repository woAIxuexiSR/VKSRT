#pragma once

#include "project_config.h"

#include <string>
#include <string_view>

inline std::string shaderPath(std::string_view rel)
{
    std::string out;
    out.reserve(sizeof(SHADER_ROOT) + 1 + rel.size());
    out.append(SHADER_ROOT).append("/").append(rel);
    return out;
}
