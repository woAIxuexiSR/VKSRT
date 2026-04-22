# VKSRT — Vulkan Ray Tracing Renderer

## 项目概述
基于 Vulkan 的光线追踪渲染器，使用链状 RenderGraph 架构。

## 参考项目
- **EasyVulkan**: `D:\works\Vscode\EasyVulkan`
- 代码风格、Vulkan 封装、CMake 结构均参考 EasyVulkan
- backend/ 和 model/ 中的代码应尽量复用/改编

## 技术栈
- **语言**: C++17
- **构建**: CMake
- **依赖管理**: vcpkg（已安装好）
- **Shader**: 全部使用 Slang（不用 GLSL），slangc 编译
- **依赖库**: Vulkan, GLFW3, GLM, stb, assimp, nlohmann_json
- **ImGui**: 从本地路径构建 `D:/software/imgui/`

## 架构设计

### 目录分工
- `src/core/` — Vulkan 核心抽象（device / swap_chain / window / resource / pipeline），从 EasyVulkan backend 移植。
- `src/passes/` — 渲染 Pass。每个 Pass 一个子目录，同目录存放其 `.slang` shader（shader 与 Pass 代码就近维护）。顶层 `pass_base.h/cpp` 提供基类、`RenderPassFactory`、`REGISTER_PASS` 宏。
- `src/neural/` — 模块化神经网络（GPU compute：Encoding 基类 + EncodingFactory + 若干 encoding 实现 + MLP + NeuralNetwork 编排器）。详见 `src/neural/README.md`。
- `src/scene/` — 场景管理（camera、gbuffer、ray_tracing_model、scene_loader）。
- `src/gui/` — ImGui 集成。
- `src/main.cpp` — 程序入口。

Shader 编译：CMake 从 `src/passes/` 子目录和 `src/shaders/neural/` 收集 `.slang`，编译到 `build/shaders/`。passes 编译为独立静态库，用 `/WHOLEARCHIVE` 链接以保留自注册静态初始化。

### 链状 RenderPass
- **工厂模式**: RenderPassFactory + REGISTER_PASS 自注册宏，main.cpp 通过 factory 创建 pass
- **图像传递**: Pass 间通过 PassImageSlot（ImageResource* + VkImageLayout）链式传递 VkImage
- **执行循环**: main.cpp 用通用 pass vector 循环执行 update → recordCommand
- 光栅化 Pass：使用 Dynamic Rendering (vkCmdBeginRendering)
- RT Pass / Compute Pass：直接写 storage image，不用 Dynamic Rendering

### Pipeline 简化
- 与 EasyVulkan 不同，Pipeline 只接收单个 slang 文件
- slang 文件包含该 Pass 所需的所有 entry point

## 里程碑

- **M1** [已完成] — 项目搭建：CMake + vcpkg、Vulkan 核心封装、ImGui、空窗口。
- **M2** [已完成] — 光栅化三角形：Pass 框架 + Dynamic Rendering。
- **M3** [已完成] — 光追 Cornell Box（简单着色）：BLAS/TLAS、RT Pass（单文件多 entry）、Blit、ACES Tonemap、相机与多帧累积重置、工厂化 Pass + PassImageSlot 链式传递。
- **M4** [已完成] — Path Tracing：独立 Accumulate → 合并进 TAA、完整 PT（多弹射 + RR + BRDF 采样）、ray_tracing 降级为可视化调试、Cornell Box 场景 + 截图。
- **M5** [已完成] — 高级渲染：NEE + MIS、双边滤波、TAA。
- **M6** [已完成] — 其他积分方式：Light Tracer（atomic splat）、Wavefront PT（6 compute kernel + atomic compaction + indirect dispatch）。
- **M7** [已完成] — Stylized：Branch MC、GGX microfacet BRDF、离线模式（--offline + PNG）、材质类型封装。
- **M8** [已完成] — GPU 神经网络：MLP forward/backward/Adam、5 种 Encoding（HashGrid / Frequency / SH / OneBlob / Identity）+ 自注册工厂、NeuralNetwork 编排器、network_test pass、NRC（嵌入 PT）。

## Build Commands
```bash
cmake -B build
cmake --build build
cmake --build build --target shaders
```

## 代码风格
与 EasyVulkan 保持一致，阅读 EasyVulkan 的 src/backend/ 和 src/model/ 了解风格。
