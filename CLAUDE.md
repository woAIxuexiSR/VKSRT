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

### 目录结构
```
src/
├── core/           # Vulkan 核心抽象（从 EasyVulkan backend 移植）
│   ├── device.h/cpp
│   ├── swap_chain.h/cpp
│   ├── window.h/cpp
│   ├── resource.h/cpp
│   └── pipeline.h/cpp
├── passes/         # 渲染 Pass（每个 Pass 一个子文件夹，shader 放同目录）
│   ├── pass_base.h/cpp   # PassBase 基类 + RenderPassFactory + REGISTER_PASS 宏
│   ├── path_tracing/     # Path Tracing（多次弹射、NEE、MIS、G-buffer 写入）
│   │   ├── path_tracing_pass.h/cpp
│   │   └── path_tracing.slang  # 单文件 3 entry points (raygen/miss/closesthit)
│   ├── taa/              # TAA（内置 accumulate 模式，reprojection + neighborhood clamping）
│   │   ├── taa_pass.h/cpp
│   │   └── taa.slang
│   ├── bilateral/        # 双边滤波降噪（基于 G-buffer 的边缘保持滤波）
│   │   ├── bilateral_pass.h/cpp
│   │   └── bilateral.slang
│   ├── tonemap/          # ACES 色调映射
│   │   ├── tonemap_pass.h/cpp
│   │   └── tonemap.slang
│   ├── blit/             # 全屏 blit，将 RT 结果绘制到 swapchain
│   │   ├── blit_pass.h/cpp
│   │   └── blit.slang
│   ├── ray_tracing/      # 可视化调试工具（Material/Position/Normal/UV 模式）
│   │   ├── ray_tracing_pass.h/cpp
│   │   └── ray_tracing.slang
│   ├── light_tracing/    # Light Tracing（光子发射 + splat + compose）
│   │   ├── light_tracing_pass.h/cpp
│   │   ├── light_tracing.slang
│   │   └── lt_compose.slang
│   ├── wavefront_pt/     # Wavefront Path Tracing（阶段式 compute，ray query）
│   │   ├── wavefront_pt_pass.h/cpp
│   │   ├── wf_generate.slang
│   │   ├── wf_prepare_indirect.slang
│   │   ├── wf_extend.slang
│   │   ├── wf_shade.slang
│   │   ├── wf_shadow.slang
│   │   └── wf_accumulate.slang
│   └── branch_pt/        # Branch PT（树状路径分叉 + stylized rendering）
│       ├── branch_pt_pass.h/cpp
│       ├── brpt_advance.slang
│       ├── brpt_propagate.slang
│       └── brpt_accumulate.slang
├── scene/          # 场景管理
│   ├── camera.h/cpp           # 从 EasyVulkan model/camera 移植，含 prevViewProj
│   ├── gbuffer.h              # App 管理的 G-buffer（position/normal/albedo）
│   ├── ray_tracing_model.h/cpp # RT 场景（加速结构 + 几何数据 + descriptor helpers）
│   └── scene_loader.h/cpp     # 场景加载（从 JSON config 构建 RayTracingModel）
├── gui/            # ImGui 集成
│   └── imgui_renderer.h/cpp  # 从 EasyVulkan 移植
└── main.cpp
```
Shader 编译：CMake 从 `src/passes/` 子目录收集 `.slang` 文件，编译输出到 `build/shaders/`。
passes 编译为独立静态库，链接到 main。

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

### M1: 项目搭建 [已完成]
- CMake + vcpkg + 目录结构
- Vulkan 核心封装（从 EasyVulkan 移植：device, swap_chain, window, resource, pipeline）
- ImGui 集成
- 空窗口能编译运行

### M2: 光栅化三角形 [已完成]
- 实现 Pass 框架（pass_base.h 基类 + 链状执行）
- passes/triangle/ 实现三角形绘制（Dynamic Rendering）
- 验证渲染管线跑通

### M3: 光追 Cornell Box（简单着色）[已完成]
- [x] RT 场景与加速结构（ray_tracing_model.h，BLAS/TLAS）
- [x] RT Pass（ray_tracing_pass.h/cpp + ray_tracing.slang，单文件多 entry point）
- [x] Blit Pass（blit_pass.h/cpp + blit.slang，全屏三角形将 RT 结果绘制到 swapchain）
- [x] Tonemap Pass（tonemap_pass.h/cpp + tonemap.slang，ACES 色调映射）
- [x] 相机控制（camera.h/cpp，支持输入和多帧累积重置）
- [x] Pass 架构重构（工厂模式 + PassImageSlot 链式传递）

### M4: Path Tracing Cornell Box [已完成]
- [x] Accumulate Pass 解耦（独立帧累积 compute pass，后合并进 TAA Pass）
- [x] PassImageSlot 链式动态传递重构（disabled 零开销透传）
- [x] 相机 UI 解耦（Application 管理，独立 ImGui section）
- [x] 完整 Path Tracing（多次弹射、Russian Roulette、BRDF 采样）
- [x] StructuredBuffer<float3> 对齐修复
- [x] ray_tracing pass 改为可视化调试工具
- [x] Cornell Box 经典场景（两个 box + Metal 镜面墙）
- [x] 截图功能（Save Image to PNG）

### M5: 高级渲染特性 [已完成]
- [x] NEE（Next Event Estimation + MIS）
- [x] 双边滤波 Pass（Bilateral Denoise）
- [x] TAA Pass（Temporal Anti-Aliasing）

### M6: 其他积分方式 [已完成]
- [x] Light Tracer（光子发射 + float atomic splat + compose）
- [x] Wavefront Path Tracing（6 compute kernels + atomic compaction + indirect dispatch）

### M7: Stylized 渲染
- [x] Branch MC
- 卡通渲染 idea 实现
- [x] GGX microfacet BRDF（Metal GGX VNDF + Lambertian diffuse+specular）
- [x] 离线渲染模式（--offline + PNG 输出）
- [x] 材质类型封装（isDeltaBRDF/isEmissive/canTransmit）

## Build Commands
```bash
cmake -B build
cmake --build build
cmake --build build --target shaders
```

## 代码风格
与 EasyVulkan 保持一致，阅读 EasyVulkan 的 src/backend/ 和 src/model/ 了解风格。
