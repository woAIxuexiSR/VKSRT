#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <string>
#include <stdexcept>

struct InputState
{
    // framebuffer resize
    bool framebufferResized{false};
    int width{0}, height{0};

    // cursor position
    bool cursorChanged{false};
    bool firstMouse{true};
    float mouseDeltaX{0.0f}, mouseDeltaY{0.0f};
    float mouseX{0.0f}, mouseY{0.0f};

    // scroll wheel
    bool scrollChanged{false};
    float scrollDeltaX{0.0f}, scrollDeltaY{0.0f};

    // keyboard input
    bool keyboardChanged{false};
    bool keyQ{false}, keyW{false}, keyE{false};
    bool keyA{false}, keyS{false}, keyD{false};

    bool isChanged() const { return framebufferResized || cursorChanged || scrollChanged || keyboardChanged; }
};

class Window
{
private:
    std::string title;
    GLFWwindow *window;
    InputState inputState;

    static void framebufferSizeCallback(GLFWwindow *window, int width, int height);
    static void cursorPositionCallback(GLFWwindow *window, double xpos, double ypos);
    static void scrollCallback(GLFWwindow *window, double xoffset, double yoffset);
    static void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods);

public:
    Window(int _w, int _h, const std::string &_title, bool visible = true);
    ~Window();

    Window(const Window &) = delete;
    Window &operator=(const Window &) = delete;

    bool shouldClose() { return glfwWindowShouldClose(window); }
    void pollEvents() { glfwPollEvents(); }
    GLFWwindow *getWindow() const { return window; }

    void checkIdle();
    VkExtent2D getExtent();
    void createWindowSurface(VkInstance instance, VkSurfaceKHR *surface);

    InputState &getInputState() { return inputState; }
    void resetInputState();
};
