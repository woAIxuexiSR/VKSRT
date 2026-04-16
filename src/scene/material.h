#pragma once

#include <glm/glm.hpp>

enum MaterialType : int
{
    MAT_LAMBERTIAN = 0,
    MAT_METAL = 1,
    MAT_DIELECTRIC = 2,
    MAT_EMISSIVE = 3,
};

struct Material
{
    glm::vec4 baseColor{0.8f, 0.8f, 0.8f, 1.0f};
    glm::vec4 emission{0.0f, 0.0f, 0.0f, 0.0f};
    float metallic{0.0f};
    float roughness{1.0f};
    float ior{1.5f};
    int type{MAT_LAMBERTIAN};
    int diffuseTexIndex{-1};
    int styleType{0};        // 0 = STYLE_NONE, future: 1, 2, ... for different g functions
    float styleParam0{0.0f}; // parameter for stylization function g (e.g. gamma)
    int _pad2{0};
};
// Must match src/shaders/material.slang layout exactly
static_assert(sizeof(Material) == 64, "Material must be 64 bytes to match Slang layout");

struct LightTriangle
{
    float area;          // world-space triangle surface area
    int indexOffset;     // triangle index in global index buffer (which triangle)
    int vertexOffset;    // vertex offset in global vertex buffer
    int matIndex;        // material index for emission lookup
    int instanceIndex;   // TLAS instance index (for transform lookup in NEE)
};
static_assert(sizeof(LightTriangle) == 20, "LightTriangle must be 20 bytes to match Slang layout");
