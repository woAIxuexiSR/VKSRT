#include "window.h"

void Window::framebufferSizeCallback(GLFWwindow *window, int width, int height)
{
    auto inputState = reinterpret_cast<InputState *>(glfwGetWindowUserPointer(window));
    inputState->framebufferResized = true;
    inputState->width = width;
    inputState->height = height;
}

void Window::cursorPositionCallback(GLFWwindow *window, double xpos, double ypos)
{
    auto inputState = reinterpret_cast<InputState *>(glfwGetWindowUserPointer(window));
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS)
    {
        inputState->cursorChanged = true;
        if (inputState->firstMouse)
        {
            inputState->mouseX = static_cast<float>(xpos);
            inputState->mouseY = static_cast<float>(ypos);
            inputState->firstMouse = false;
        }
        inputState->mouseDeltaX += static_cast<float>(xpos) - inputState->mouseX;
        inputState->mouseDeltaY += static_cast<float>(ypos) - inputState->mouseY;
        inputState->mouseX = static_cast<float>(xpos);
        inputState->mouseY = static_cast<float>(ypos);
    }
    else
        inputState->firstMouse = true;
}

void Window::scrollCallback(GLFWwindow *window, double xoffset, double yoffset)
{
    auto inputState = reinterpret_cast<InputState *>(glfwGetWindowUserPointer(window));
    if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS)
    {
        inputState->scrollChanged = true;
        inputState->scrollDeltaX += static_cast<float>(xoffset);
        inputState->scrollDeltaY += static_cast<float>(yoffset);
    }
}

void Window::keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods)
{
    auto inputState = reinterpret_cast<InputState *>(glfwGetWindowUserPointer(window));
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    if (key == GLFW_KEY_Q && (action == GLFW_PRESS || action == GLFW_REPEAT))
    {
        inputState->keyboardChanged = true;
        inputState->keyQ = true;
    }
    else if (key == GLFW_KEY_W && (action == GLFW_PRESS || action == GLFW_REPEAT))
    {
        inputState->keyboardChanged = true;
        inputState->keyW = true;
    }
    else if (key == GLFW_KEY_E && (action == GLFW_PRESS || action == GLFW_REPEAT))
    {
        inputState->keyboardChanged = true;
        inputState->keyE = true;
    }
    else if (key == GLFW_KEY_A && (action == GLFW_PRESS || action == GLFW_REPEAT))
    {
        inputState->keyboardChanged = true;
        inputState->keyA = true;
    }
    else if (key == GLFW_KEY_S && (action == GLFW_PRESS || action == GLFW_REPEAT))
    {
        inputState->keyboardChanged = true;
        inputState->keyS = true;
    }
    else if (key == GLFW_KEY_D && (action == GLFW_PRESS || action == GLFW_REPEAT))
    {
        inputState->keyboardChanged = true;
        inputState->keyD = true;
    }
}

Window::Window(int _w, int _h, const std::string &_title, bool visible)
    : title(_title)
{
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    if (!visible)
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

    int width = _w, height = _h;
    inputState.width = width;
    inputState.height = height;
    window = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
    if (window == nullptr)
        throw std::runtime_error("failed to create GLFW window");

    glfwSetWindowUserPointer(window, &inputState);
    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
    glfwSetCursorPosCallback(window, cursorPositionCallback);
    glfwSetScrollCallback(window, scrollCallback);
    glfwSetKeyCallback(window, keyCallback);
}

Window::~Window()
{
    glfwDestroyWindow(window);
    glfwTerminate();
}

void Window::checkIdle()
{
    while (inputState.width == 0 || inputState.height == 0)
        glfwWaitEvents();
}

VkExtent2D Window::getExtent()
{
    return {static_cast<uint32_t>(inputState.width), static_cast<uint32_t>(inputState.height)};
}

void Window::createWindowSurface(VkInstance instance, VkSurfaceKHR *surface)
{
    if (glfwCreateWindowSurface(instance, window, nullptr, surface) != VK_SUCCESS)
        throw std::runtime_error("failed to create window surface");
}

void Window::resetInputState()
{
    inputState.framebufferResized = false;

    inputState.cursorChanged = false;
    inputState.mouseDeltaX = inputState.mouseDeltaY = 0.0f;

    inputState.scrollChanged = false;
    inputState.scrollDeltaX = inputState.scrollDeltaY = 0.0f;

    inputState.keyboardChanged = false;
    inputState.keyQ = false;
    inputState.keyW = false;
    inputState.keyE = false;
    inputState.keyA = false;
    inputState.keyS = false;
    inputState.keyD = false;
}
