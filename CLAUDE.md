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
- **依赖库**: Vulkan, GLFW3, GLM, stb, assimp
- **ImGui**: 从本地路径构建 `D:/software/imgui/`

## 架构设计

### 目录结构
```
src/
├── core/           # Vulkan 核心抽象（从 EasyVulkan backend 移植）
│   ├── device.h/cpp
│   ├── swapchain.h/cpp   (EasyVulkan: swap_chain)
│   ├── window.h/cpp
│   ├── resource.h/cpp
│   ├── pipeline.h/cpp    # 简化：只传单个 slang 文件
│   └── command.h/cpp
├── passes/         # 渲染 Pass（每个 Pass 一个子文件夹，shader 放同目录）
│   ├── pass_base.h       # Pass 基类
│   ├── triangle/         # M2: 光栅化三角形
│   │   ├── triangle_pass.h/cpp
│   │   └── triangle.slang
│   ├── ray_tracing/      # M3: 光追
│   │   ├── ray_tracing_pass.h/cpp
│   │   └── ray_tracing.slang
│   └── tonemap/          # 色调映射
│       ├── tonemap_pass.h/cpp
│       └── tonemap.slang
├── scene/          # 场景管理
│   ├── camera.h/cpp      # 从 EasyVulkan model/camera 移植
│   └── cornell_box.h/cpp # 程序化生成 Cornell Box
├── gui/            # ImGui 集成
│   └── imgui_renderer.h/cpp  # 从 EasyVulkan 移植
└── main.cpp
shaders/            # CMake 会从 passes/ 子目录收集 .slang 文件编译
```

### 链状 RenderPass
- Pass 间传递 VkImage + 可选额外属性
- Pass 使用**单例模式**
- 光栅化 Pass：使用 Dynamic Rendering (vkCmdBeginRendering)
- RT Pass / Compute Pass：直接写 storage image，不用 Dynamic Rendering
- 程序输入通过 **config.json** 指定模型路径、pass 信息等

### Pipeline 简化
- 与 EasyVulkan 不同，Pipeline 只接收单个 slang 文件
- slang 文件包含该 Pass 所需的所有 entry point

## 里程碑

### M1: 项目搭建
- CMake + vcpkg + 目录结构
- Vulkan 核心封装（从 EasyVulkan 移植）
- 空窗口能编译运行

### M2: 光栅化三角形
- 实现 Pass 框架（基类 + 链状执行）
- passes/triangle/ 实现三角形绘制
- 验证渲染管线跑通

### M3: 光追 Cornell Box（简单着色）
- 程序化生成 Cornell Box
- BLAS/TLAS 加速结构
- RT Pass + Tonemap Pass
- 直接光照

### M4: Path Tracing Cornell Box
- 升级为 Path Tracing（多次弹射）
- 多帧累积（progressive rendering）
- 相机控制

## Build Commands
```bash
cmake -B build
cmake --build build
cmake --build build --target shaders
```

## 代码风格
与 EasyVulkan 保持一致，阅读 EasyVulkan 的 src/backend/ 和 src/model/ 了解风格。
