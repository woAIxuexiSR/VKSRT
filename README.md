# VKSRT

Vulkan ray-tracing renderer with a chain-style render graph. Supports path tracing, light tracing, wavefront PT, branched PT, and a neural radiance cache.

## Dependencies

- **Vulkan SDK** (for `vulkan-1` and `slangc`) — set the `VULKAN_SDK` env var.
- **CMake 3.16+**, a C++17 compiler.
- GPU + driver supporting `VK_KHR_ray_tracing_pipeline` and `VK_KHR_acceleration_structure`.
- Libraries: `glm`, `glfw3`, `stb`, `nlohmann-json`, `assimp`, `imgui` (with GLFW + Vulkan backends).

## Build

```
cmake -B build
cmake --build build
```

Shader sources (`src/**/*.slang`) are compiled to SPIR-V under `build/shaders/` as part of the build.

## Run

```
./build/main                       # loads default config.json
./build/main scenes/path_tracing.json
./build/main --help
```

Scene configs live in `scenes/` — see `scenes/README.md` for details.
