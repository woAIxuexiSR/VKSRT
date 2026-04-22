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

// Directory of the currently-loaded scene config (absolute path, no trailing slash).
// main.cpp sets this right after parsing the config file; modules use configPath()
// to resolve paths specified as relative in the config.
inline std::string &configDirRef()
{
    static std::string dir;
    return dir;
}

inline void setConfigDir(std::string_view dir) { configDirRef() = dir; }
inline const std::string &getConfigDir() { return configDirRef(); }

// Resolve a path read from the scene config:
//   - empty input -> empty output
//   - absolute input -> returned as-is
//   - relative input -> joined against configDir
inline std::string configRelPath(std::string_view rel)
{
    if (rel.empty())
        return {};

    const bool isAbsolute =
        (rel.size() >= 1 && (rel[0] == '/' || rel[0] == '\\')) ||
        (rel.size() >= 2 && rel[1] == ':');
    if (isAbsolute)
        return std::string(rel);

    const std::string &base = configDirRef();
    if (base.empty())
        return std::string(rel);

    std::string out;
    out.reserve(base.size() + 1 + rel.size());
    out.append(base).append("/").append(rel);
    return out;
}
