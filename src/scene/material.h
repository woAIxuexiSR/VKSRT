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
    int _pad0{0};
    int _pad1{0};
    int _pad2{0};
};
// sizeof(Material) = 64 bytes
