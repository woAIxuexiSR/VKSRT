#pragma once

#include "ray_tracing_model.h"
#include <nlohmann/json.hpp>
#include <string>

using json = nlohmann::json;

class SceneLoader
{
public:
    // basePath: directory to resolve relative model paths against (typically config file's directory).
    static void loadScene(const json &scene, RayTracingModel &model, const std::string &basePath = "");

    static void loadModel(const std::string &path, RayTracingModel &model,
                          const glm::mat4 &transform = glm::mat4(1.0f),
                          const json &materialOverride = json{});

    static void buildCornellBox(RayTracingModel &model);

    static glm::mat4 computeTRSMatrix(glm::vec3 translation, glm::vec3 rotationDeg, glm::vec3 scale);
    static void applyMaterialOverride(Material &mat, const json &override_json);
};
