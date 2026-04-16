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
                          const glm::mat4 &transform = glm::mat4(1.0f),
                          const json &materialOverride = json{});

    static void buildCornellBox(RayTracingModel &model);

    static glm::mat4 computeTRSMatrix(glm::vec3 translation, glm::vec3 rotationDeg, glm::vec3 scale);
    static void applyMaterialOverride(Material &mat, const json &override_json);
};
