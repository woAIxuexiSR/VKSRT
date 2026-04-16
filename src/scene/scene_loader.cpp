#include "scene_loader.h"

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <glm/gtc/matrix_transform.hpp>

#include <cmath>
#include <iostream>
#include <stdexcept>

// --- Material type string parsing ---

static int parseMaterialType(const std::string &s)
{
    if (s == "lambertian") return MAT_LAMBERTIAN;
    if (s == "metal") return MAT_METAL;
    if (s == "dielectric") return MAT_DIELECTRIC;
    if (s == "emissive") return MAT_EMISSIVE;
    return -1;
}

// --- TRS matrix ---

glm::mat4 SceneLoader::computeTRSMatrix(glm::vec3 translation, glm::vec3 rotationDeg, glm::vec3 scale)
{
    glm::mat4 T = glm::translate(glm::mat4(1.0f), translation);
    glm::mat4 Rx = glm::rotate(glm::mat4(1.0f), glm::radians(rotationDeg.x), glm::vec3(1, 0, 0));
    glm::mat4 Ry = glm::rotate(glm::mat4(1.0f), glm::radians(rotationDeg.y), glm::vec3(0, 1, 0));
    glm::mat4 Rz = glm::rotate(glm::mat4(1.0f), glm::radians(rotationDeg.z), glm::vec3(0, 0, 1));
    glm::mat4 S = glm::scale(glm::mat4(1.0f), scale);
    return T * Rz * Ry * Rx * S;
}

// --- Material override from JSON ---

void SceneLoader::applyMaterialOverride(Material &mat, const json &ov)
{
    if (ov.is_null() || ov.empty())
        return;

    if (ov.contains("type"))
    {
        int t = parseMaterialType(ov["type"].get<std::string>());
        if (t >= 0)
            mat.type = t;
    }
    if (ov.contains("roughness"))
        mat.roughness = ov["roughness"].get<float>();
    if (ov.contains("metallic"))
        mat.metallic = ov["metallic"].get<float>();
    if (ov.contains("ior"))
        mat.ior = ov["ior"].get<float>();
    if (ov.contains("styleType"))
        mat.styleType = ov["styleType"].get<int>();
    if (ov.contains("styleParam0"))
        mat.styleParam0 = ov["styleParam0"].get<float>();
    if (ov.contains("baseColor"))
    {
        auto &c = ov["baseColor"];
        float a = c.size() > 3 ? c[3].get<float>() : 1.0f;
        mat.baseColor = {c[0].get<float>(), c[1].get<float>(), c[2].get<float>(), a};
    }
    if (ov.contains("emission"))
    {
        auto &e = ov["emission"];
        float a = e.size() > 3 ? e[3].get<float>() : glm::max(e[0].get<float>(), glm::max(e[1].get<float>(), e[2].get<float>()));
        mat.emission = {e[0].get<float>(), e[1].get<float>(), e[2].get<float>(), a};
        if (a > 0.0f)
            mat.type = MAT_EMISSIVE;
    }
}

// --- JSON helpers ---

static glm::vec3 parseVec3(const json &j, glm::vec3 def = {0, 0, 0})
{
    if (j.is_array() && j.size() >= 3)
        return {j[0].get<float>(), j[1].get<float>(), j[2].get<float>()};
    return def;
}

static glm::vec3 parseScale(const json &j, glm::vec3 def = {1, 1, 1})
{
    if (j.is_number())
    {
        float s = j.get<float>();
        return {s, s, s};
    }
    if (j.is_array() && j.size() >= 3)
        return {j[0].get<float>(), j[1].get<float>(), j[2].get<float>()};
    return def;
}

// --- Scene loading ---

void SceneLoader::loadScene(const json &params, RayTracingModel &model)
{
    if (!params.contains("scene"))
    {
        buildCornellBox(model);
        return;
    }

    auto &scene = params["scene"];
    std::string type = scene.value("type", "cornell_box");

    if (type == "models")
    {
        if (!scene.contains("models") || !scene["models"].is_array())
            throw std::runtime_error("SceneLoader: 'models' type requires a 'models' array");

        for (auto &entry : scene["models"])
        {
            std::string path = entry.at("path").get<std::string>();
            glm::vec3 translation = entry.contains("translation") ? parseVec3(entry["translation"]) : glm::vec3(0);
            glm::vec3 rotation = entry.contains("rotation") ? parseVec3(entry["rotation"]) : glm::vec3(0);
            glm::vec3 scale = entry.contains("scale") ? parseScale(entry["scale"]) : glm::vec3(1);

            glm::mat4 transform = computeTRSMatrix(translation, rotation, scale);
            json matOverride = entry.contains("material") ? entry["material"] : json{};

            loadModel(path, model, transform, matOverride);
        }
    }
    else
    {
        buildCornellBox(model);
    }
}

void SceneLoader::loadModel(const std::string &path, RayTracingModel &model,
                            const glm::mat4 &transform, const json &materialOverride)
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

        // Store vertices in object space (transform applied via TLAS instance)
        std::vector<glm::vec3> vertices(mesh->mNumVertices);
        std::vector<glm::vec3> normals(mesh->mNumVertices);
        std::vector<glm::vec2> texcoords(mesh->mNumVertices, glm::vec2(0.0f));

        for (unsigned int i = 0; i < mesh->mNumVertices; i++)
        {
            vertices[i] = glm::vec3(mesh->mVertices[i].x, mesh->mVertices[i].y, mesh->mVertices[i].z);
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

        applyMaterialOverride(mat, materialOverride);
        model.insertMesh(vertices, indices, mat, normals, texcoords, transform);
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

// Helper: add a box from 6 quads given 8 corner vertices (axis-aligned before rotation)
// Vertices are specified as: bottom face (b0-b3), top face (t0-t3), wound CCW from outside
static void addBox(RayTracingModel &model,
                   glm::vec3 b0, glm::vec3 b1, glm::vec3 b2, glm::vec3 b3,
                   glm::vec3 t0, glm::vec3 t1, glm::vec3 t2, glm::vec3 t3,
                   const Material &mat)
{
    // Bottom (normal pointing down)
    addQuad(model, b0, b3, b2, b1, mat);
    // Top (normal pointing up)
    addQuad(model, t0, t1, t2, t3, mat);
    // Front
    addQuad(model, b0, b1, t1, t0, mat);
    // Right
    addQuad(model, b1, b2, t2, t1, mat);
    // Back
    addQuad(model, b2, b3, t3, t2, mat);
    // Left
    addQuad(model, b3, b0, t0, t3, mat);
}

// Rotate a point around the Z axis by angle (radians) relative to a center
static glm::vec3 rotateZ(glm::vec3 p, glm::vec3 center, float angle)
{
    float c = std::cos(angle);
    float s = std::sin(angle);
    float dx = p.x - center.x;
    float dy = p.y - center.y;
    return {center.x + dx * c - dy * s,
            center.y + dx * s + dy * c,
            p.z};
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

    Material mirror{};
    mirror.baseColor = {0.95f, 0.7f, 0.3f, 1.0f}; // gold
    mirror.roughness = 0.15f;
    mirror.metallic = 1.0f;
    mirror.type = MAT_METAL;

    Material glass{};
    glass.baseColor = {1.0f, 1.0f, 1.0f, 1.0f};
    glass.roughness = 0.0f;
    glass.ior = 1.5f;
    glass.type = MAT_DIELECTRIC;

    Material light{};
    light.baseColor = {1.0f, 1.0f, 1.0f, 1.0f};
    light.emission = {1.0f, 1.0f, 1.0f, 15.0f};
    light.type = MAT_EMISSIVE;

    // Z-up coordinate system: floor z=0, ceiling z=1, open face y=0 (camera looks in from -Y)
    // Floor (z=0), normal +Z
    addQuad(model,
            {0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, white);
    // Ceiling (z=1), normal -Z
    addQuad(model,
            {0, 0, 1}, {0, 1, 1}, {1, 1, 1}, {1, 0, 1}, white);
    // Back wall (y=1), normal -Y
    addQuad(model,
            {0, 1, 0}, {1, 1, 0}, {1, 1, 1}, {0, 1, 1}, white);
    // Left wall - red (x=0), normal +X
    addQuad(model,
            {0, 0, 0}, {0, 1, 0}, {0, 1, 1}, {0, 0, 1}, red);
    // Right wall - metal mirror (x=1), normal -X
    addQuad(model,
            {1, 1, 0}, {1, 0, 0}, {1, 0, 1}, {1, 1, 1}, mirror);
    // Light (ceiling, slightly inset), normal -Z
    addQuad(model,
            {0.35f, 0.35f, 0.999f}, {0.35f, 0.65f, 0.999f},
            {0.65f, 0.65f, 0.999f}, {0.65f, 0.35f, 0.999f}, light);

    // --- Tall box (right-back): white lambertian ---
    // Classic Cornell Box tall box: ~0.3 wide, ~0.6 tall, rotated ~15 deg
    {
        float bx = 0.62f, by = 0.55f; // center
        float hw = 0.15f;              // half-width
        float h = 0.6f;
        float angle = glm::radians(15.0f);
        glm::vec3 center = {bx, by, 0};

        glm::vec3 b0 = rotateZ({bx - hw, by - hw, 0}, center, angle);
        glm::vec3 b1 = rotateZ({bx + hw, by - hw, 0}, center, angle);
        glm::vec3 b2 = rotateZ({bx + hw, by + hw, 0}, center, angle);
        glm::vec3 b3 = rotateZ({bx - hw, by + hw, 0}, center, angle);
        glm::vec3 t0 = b0; t0.z = h;
        glm::vec3 t1 = b1; t1.z = h;
        glm::vec3 t2 = b2; t2.z = h;
        glm::vec3 t3 = b3; t3.z = h;

        addBox(model, b0, b1, b2, b3, t0, t1, t2, t3, white);
    }

    // --- Short box (left-front): dielectric glass ---
    // Classic Cornell Box short box: ~0.3 wide, ~0.3 tall, rotated ~-18 deg
    {
        float bx = 0.35f, by = 0.3f; // center
        float hw = 0.15f;             // half-width
        float h = 0.3f;
        float angle = glm::radians(-18.0f);
        glm::vec3 center = {bx, by, 0};

        glm::vec3 b0 = rotateZ({bx - hw, by - hw, 0}, center, angle);
        glm::vec3 b1 = rotateZ({bx + hw, by - hw, 0}, center, angle);
        glm::vec3 b2 = rotateZ({bx + hw, by + hw, 0}, center, angle);
        glm::vec3 b3 = rotateZ({bx - hw, by + hw, 0}, center, angle);
        glm::vec3 t0 = b0; t0.z = h;
        glm::vec3 t1 = b1; t1.z = h;
        glm::vec3 t2 = b2; t2.z = h;
        glm::vec3 t3 = b3; t3.z = h;

        addBox(model, b0, b1, b2, b3, t0, t1, t2, t3, glass);
    }
}
