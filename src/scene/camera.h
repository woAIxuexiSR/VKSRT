#pragma once

#include "window.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <algorithm>

class Camera
{
public:
    glm::vec3 pos, target;
    glm::vec3 front, up, right;
    float aspect, fov, near, far;
    float yaw, pitch;

    static constexpr glm::vec3 worldUp = {0.0f, 0.0f, 1.0f};
    static constexpr glm::mat4 flipY = {1.0f, 0.0f, 0.0f, 0.0f,
                                        0.0f, -1.0f, 0.0f, 0.0f,
                                        0.0f, 0.0f, 1.0f, 0.0f,
                                        0.0f, 0.0f, 0.0f, 1.0f};
    static constexpr float moveSpeed = 3.0f;
    static constexpr float sensitivity = 0.1f;

public:
    Camera(glm::vec3 _p, glm::vec3 _t, float _aspect, float _fov, float _n = 0.1f, float _f = 100.0f);
    ~Camera() {}

    void processInput(InputState &inputState, float deltaTime);

    glm::mat4 getViewMatrix() const;
    glm::mat4 getProjectionMatrix() const;
    glm::mat4 getInverseViewMatrix() const;
    glm::mat4 getInverseProjectionMatrix() const;
};
