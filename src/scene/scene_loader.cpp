#include "scene_loader.h"

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <iostream>
#include <stdexcept>

void SceneLoader::loadModel(const std::string &path, RayTracingModel &model,
                            float scale, glm::vec3 offset)
{
    Assimp::Importer importer;
    const aiScene *scene = importer.ReadFile(path,
                                             aiProcess_Triangulate |
                                                 aiProcess_GenNormals |
                                                 aiProcess_FlipUVs |
                                                 aiProcess_PreTransformVertices);

    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
        throw std::runtime_error("SceneLoader: failed to load '" + path + "': " + importer.GetErrorString());

    std::cout << "Loading model: " << path << " (" << scene->mNumMeshes << " meshes)" << std::endl;

    for (unsigned int m = 0; m < scene->mNumMeshes; m++)
    {
        aiMesh *mesh = scene->mMeshes[m];

        std::vector<glm::vec3> vertices(mesh->mNumVertices);
        std::vector<glm::vec3> normals(mesh->mNumVertices);
        std::vector<glm::vec2> texcoords(mesh->mNumVertices, glm::vec2(0.0f));

        for (unsigned int i = 0; i < mesh->mNumVertices; i++)
        {
            vertices[i] = glm::vec3(mesh->mVertices[i].x, mesh->mVertices[i].y, mesh->mVertices[i].z) * scale + offset;
            normals[i] = glm::vec3(mesh->mNormals[i].x, mesh->mNormals[i].y, mesh->mNormals[i].z);
            if (mesh->mTextureCoords[0])
                texcoords[i] = glm::vec2(mesh->mTextureCoords[0][i].x, mesh->mTextureCoords[0][i].y);
        }

        std::vector<glm::uvec3> indices;
        indices.reserve(mesh->mNumFaces);
        for (unsigned int i = 0; i < mesh->mNumFaces; i++)
        {
            aiFace &face = mesh->mFaces[i];
            if (face.mNumIndices == 3)
                indices.push_back({face.mIndices[0], face.mIndices[1], face.mIndices[2]});
        }

        Material mat{};
        if (mesh->mMaterialIndex < scene->mNumMaterials)
        {
            aiMaterial *aiMat = scene->mMaterials[mesh->mMaterialIndex];

            aiColor4D diffuse;
            if (aiMat->Get(AI_MATKEY_COLOR_DIFFUSE, diffuse) == AI_SUCCESS)
                mat.baseColor = glm::vec4(diffuse.r, diffuse.g, diffuse.b, diffuse.a);

            aiColor3D emissive;
            if (aiMat->Get(AI_MATKEY_COLOR_EMISSIVE, emissive) == AI_SUCCESS)
            {
                float emStrength = glm::max(emissive.r, glm::max(emissive.g, emissive.b));
                if (emStrength > 0.0f)
                {
                    mat.emission = glm::vec4(emissive.r, emissive.g, emissive.b, emStrength);
                    mat.type = MAT_EMISSIVE;
                }
            }

            float metallic = 0.0f;
            if (aiMat->Get(AI_MATKEY_METALLIC_FACTOR, metallic) == AI_SUCCESS)
                mat.metallic = metallic;

            float roughness = 1.0f;
            if (aiMat->Get(AI_MATKEY_ROUGHNESS_FACTOR, roughness) == AI_SUCCESS)
                mat.roughness = roughness;
        }

        model.insertMesh(vertices, indices, mat, normals, texcoords);
    }
}

static void addQuad(RayTracingModel &model,
                    glm::vec3 a, glm::vec3 b, glm::vec3 c, glm::vec3 d,
                    const Material &mat)
{
    glm::vec3 normal = glm::normalize(glm::cross(b - a, c - a));
    std::vector<glm::vec3> verts = {a, b, c, d};
    std::vector<glm::vec3> norms(4, normal);
    std::vector<glm::vec2> uvs = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
    std::vector<glm::uvec3> indices = {{0, 1, 2}, {0, 2, 3}};
    model.insertMesh(verts, indices, mat, norms, uvs);
}

void SceneLoader::buildCornellBox(RayTracingModel &model)
{
    Material white{};
    white.baseColor = {0.73f, 0.73f, 0.73f, 1.0f};
    white.roughness = 1.0f;

    Material red{};
    red.baseColor = {0.65f, 0.05f, 0.05f, 1.0f};
    red.roughness = 1.0f;

    Material green{};
    green.baseColor = {0.12f, 0.45f, 0.15f, 1.0f};
    green.roughness = 1.0f;

    Material light{};
    light.baseColor = {1.0f, 1.0f, 1.0f, 1.0f};
    light.emission = {1.0f, 1.0f, 1.0f, 15.0f};
    light.type = MAT_EMISSIVE;

    // Floor (y=0), normal +Y
    addQuad(model,
            {0, 0, 0}, {0, 0, 1}, {1, 0, 1}, {1, 0, 0}, white);
    // Ceiling (y=1), normal -Y
    addQuad(model,
            {0, 1, 0}, {1, 1, 0}, {1, 1, 1}, {0, 1, 1}, white);
    // Back wall (z=1), normal -Z
    addQuad(model,
            {0, 0, 1}, {0, 1, 1}, {1, 1, 1}, {1, 0, 1}, white);
    // Left wall - red (x=0), normal +X
    addQuad(model,
            {0, 0, 0}, {0, 1, 0}, {0, 1, 1}, {0, 0, 1}, red);
    // Right wall - green (x=1), normal -X
    addQuad(model,
            {1, 0, 0}, {1, 0, 1}, {1, 1, 1}, {1, 1, 0}, green);
    // Light (ceiling, slightly inset), normal -Y
    addQuad(model,
            {0.35f, 0.999f, 0.35f}, {0.65f, 0.999f, 0.35f},
            {0.65f, 0.999f, 0.65f}, {0.35f, 0.999f, 0.65f}, light);
}
