#pragma once

#include "ray_tracing_model.h"
#include <nlohmann/json.hpp>
#include <string>

using json = nlohmann::json;

class SceneLoader
{
public:
    static void loadScene(const json &params, RayTracingModel &model);

    static void loadModel(const std::string &path, RayTracingModel &model,
                          float scale = 1.0f, glm::vec3 offset = {0, 0, 0});

    static void buildCornellBox(RayTracingModel &model);
};
