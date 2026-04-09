#include "camera.h"

Camera::Camera(glm::vec3 _p, glm::vec3 _t, float _aspect, float _fov, float _n, float _f)
    : pos(_p), target(_t), aspect(_aspect), fov(_fov), near(_n), far(_f), yaw(0.0f), pitch(0.0f)
{
    front = glm::normalize(target - pos);
    right = glm::normalize(glm::cross(front, worldUp));
    up = glm::normalize(glm::cross(right, front));
}

void Camera::processInput(InputState &inputState, float deltaTime)
{
    if (inputState.framebufferResized)
        aspect = (float)inputState.width / (float)inputState.height;

    if (inputState.cursorChanged)
    {
        yaw += inputState.mouseDeltaX * sensitivity;
        pitch = std::clamp(pitch + inputState.mouseDeltaY * sensitivity, -89.0f, 89.0f);
    }

    if (inputState.scrollChanged)
        fov = std::clamp(fov - inputState.scrollDeltaY, 1.0f, 90.0f);

    if (inputState.keyboardChanged)
    {
        glm::vec3 changed = glm::vec3(0.0f);
        if (inputState.keyQ)
            changed += front * moveSpeed * deltaTime;
        if (inputState.keyW)
            changed += worldUp * moveSpeed * deltaTime;
        if (inputState.keyE)
            changed -= front * moveSpeed * deltaTime;
        if (inputState.keyA)
            changed -= right * moveSpeed * deltaTime;
        if (inputState.keyS)
            changed -= worldUp * moveSpeed * deltaTime;
        if (inputState.keyD)
            changed += right * moveSpeed * deltaTime;

        pos += changed;
        target += changed;
    }
}

glm::mat4 Camera::getViewMatrix() const
{
    auto view = glm::lookAt(pos, target, up);
    auto rotX = glm::rotate(view, glm::radians(yaw), up);
    auto rotY = glm::rotate(rotX, glm::radians(pitch), right);
    return rotY;
}

glm::mat4 Camera::getProjectionMatrix() const
{
    return flipY * glm::perspective(glm::radians(fov), aspect, near, far);
}

glm::mat4 Camera::getInverseViewMatrix() const
{
    return glm::inverse(getViewMatrix());
}

glm::mat4 Camera::getInverseProjectionMatrix() const
{
    return glm::inverse(getProjectionMatrix());
}
