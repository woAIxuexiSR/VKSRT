#pragma once

#include "ray_tracing_model.h"
#include <string>

class SceneLoader
{
public:
    static void loadModel(const std::string &path, RayTracingModel &model,
                          float scale = 1.0f, glm::vec3 offset = {0, 0, 0});

    static void buildCornellBox(RayTracingModel &model);
};
