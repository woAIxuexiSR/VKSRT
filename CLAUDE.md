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
│   ├── swap_chain.h/cpp
│   ├── window.h/cpp
│   ├── resource.h/cpp
│   └── pipeline.h/cpp    # 简化：只传单个 slang 文件
├── passes/         # 渲染 Pass（每个 Pass 一个子文件夹，shader 放同目录）
│   ├── pass_base.h       # Pass 基类
│   ├── triangle/         # M2: 光栅化三角形
│   │   ├── triangle_pass.h  (header-only)
│   │   └── triangle.slang
│   ├── ray_tracing/      # M3: 光追
│   │   ├── ray_tracing_pass.h  (header-only)
│   │   ├── ray_tracing.slang
│   │   └── blit.slang    # 全屏 blit，将 RT 结果绘制到 swapchain
│   └── tonemap/          # 色调映射（待实现）
│       ├── tonemap_pass.h
│       └── tonemap.slang
├── scene/          # 场景管理
│   ├── camera.h/cpp      # 从 EasyVulkan model/camera 移植
│   └── cornell_box.h     # 程序化生成 Cornell Box (header-only)
├── gui/            # ImGui 集成
│   └── imgui_renderer.h/cpp  # 从 EasyVulkan 移植
└── main.cpp
```
Shader 编译：CMake 从 `src/passes/` 子目录收集 `.slang` 文件，编译输出到 `build/shaders/`。

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

### M1: 项目搭建 [已完成]
- CMake + vcpkg + 目录结构
- Vulkan 核心封装（从 EasyVulkan 移植：device, swap_chain, window, resource, pipeline）
- ImGui 集成
- 空窗口能编译运行

### M2: 光栅化三角形 [已完成]
- 实现 Pass 框架（pass_base.h 基类 + 链状执行）
- passes/triangle/ 实现三角形绘制（Dynamic Rendering）
- 验证渲染管线跑通

### M3: 光追 Cornell Box（简单着色）[进行中]
- [x] 程序化生成 Cornell Box（cornell_box.h）
- [x] BLAS/TLAS 加速结构
- [x] RT Pass（ray_tracing_pass.h + ray_tracing.slang）
- [x] Blit Pass（blit.slang，全屏三角形将 RT 结果绘制到 swapchain）
- [x] 相机控制（camera.h/cpp，支持输入和多帧累积重置）
- [ ] Tonemap Pass（待实现）

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
