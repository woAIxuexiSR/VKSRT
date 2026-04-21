cmake_minimum_required(VERSION 3.16)

project(VKSRT LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

# dependencies via vcpkg
find_package(glm CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(Vulkan REQUIRED)
find_package(Stb REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(assimp CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)

# compile slang shaders
set(SPIRV_BINARY_FILES "")
if(NOT SLANGC_EXECUTABLE)
    if(DEFINED ENV{VULKAN_SDK})
        set(SLANGC_EXECUTABLE "$ENV{VULKAN_SDK}/bin/slangc")
    else()
        message(FATAL_ERROR "slangc not found")
    endif()
endif()

# Shared shader modules (import-only, not compiled to SPV independently)
set(SLANG_SHARED_DIR "${CMAKE_SOURCE_DIR}/src/shaders")
set(SLANG_NEURAL_DIR "${CMAKE_SOURCE_DIR}/src/shaders/neural")
file(GLOB SLANG_SHARED_MODULES CONFIGURE_DEPENDS "${SLANG_SHARED_DIR}/*.slang")
file(GLOB SLANG_NEURAL_MODULES CONFIGURE_DEPENDS "${SLANG_NEURAL_DIR}/*.slang")

# Helper: compile a single .slang file to .spv.
#   SLANG_FILE:   absolute path to input .slang
#   OUT_SPV:      absolute path to output .spv
#   ENTRIES:      optional list of entry point names (empty = slangc default single entry)
#   PROFILE:      optional slangc profile (e.g. glsl_460). Omit for RT shaders - slangc
#                 will infer capabilities from entry points; forcing glsl_460 on RT
#                 shaders drops capabilities silently and produces broken SPV.
# Output is appended to SPIRV_BINARY_FILES in the caller's scope.
function(compile_slang_shader SLANG_FILE OUT_SPV)
    set(oneValueArgs PROFILE)
    set(multiValueArgs ENTRIES)
    cmake_parse_arguments(SS "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    get_filename_component(SLANG_DIR ${SLANG_FILE} DIRECTORY)
    get_filename_component(SLANG_NAME ${SLANG_FILE} NAME)
    get_filename_component(OUT_DIR ${OUT_SPV} DIRECTORY)

    set(ENTRY_FLAGS "")
    foreach(E ${SS_ENTRIES})
        list(APPEND ENTRY_FLAGS -entry ${E})
    endforeach()

    set(PROFILE_FLAGS "")
    if(SS_PROFILE)
        set(PROFILE_FLAGS -profile ${SS_PROFILE})
    endif()

    add_custom_command(
        OUTPUT ${OUT_SPV}
        COMMAND ${CMAKE_COMMAND} -E make_directory "${OUT_DIR}"
        COMMAND ${SLANGC_EXECUTABLE} ${SLANG_FILE}
                -I${SLANG_DIR} -I${SLANG_SHARED_DIR} -I${SLANG_NEURAL_DIR}
                ${PROFILE_FLAGS} -target spirv
                ${ENTRY_FLAGS}
                -o ${OUT_SPV}
        DEPENDS ${SLANG_FILE} ${SLANG_SHARED_MODULES} ${SLANG_NEURAL_MODULES}
        COMMENT "Compiling ${SLANG_NAME} to SPIR-V"
    )
    set(SPIRV_BINARY_FILES ${SPIRV_BINARY_FILES} ${OUT_SPV} PARENT_SCOPE)
endfunction()

# RT slang shaders: multi-entry point files compiled to a single SPV each.
# (Vulkan ray-tracing pipelines need raygen/miss/closesthit bound together.)
set(RT_MULTI_ENTRY_SHADERS
    "ray_tracing/ray_tracing"
    "path_tracing/path_tracing"
    "light_tracing/light_tracing"
    "nrc/nrc"
)
# Build an absolute-path set for quick skip in the single-entry loop below.
set(RT_MULTI_ENTRY_ABS "")
foreach(RT_SHADER ${RT_MULTI_ENTRY_SHADERS})
    set(RT_SLANG "${CMAKE_SOURCE_DIR}/src/passes/${RT_SHADER}.slang")
    set(RT_SPV   "${CMAKE_BINARY_DIR}/shaders/${RT_SHADER}.spv")
    list(APPEND RT_MULTI_ENTRY_ABS ${RT_SLANG})
    compile_slang_shader(${RT_SLANG} ${RT_SPV}
        ENTRIES raygenMain missMain closestHitMain)
endforeach()

# Single-entry pass shaders: glob everything under src/passes, skip RT multi-entry files.
file(GLOB_RECURSE SLANG_FILES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/passes/*.slang")
foreach(SLANG ${SLANG_FILES})
    list(FIND RT_MULTI_ENTRY_ABS ${SLANG} IDX)
    if(NOT IDX EQUAL -1)
        continue()
    endif()
    file(RELATIVE_PATH REL_PATH ${CMAKE_SOURCE_DIR}/src/passes ${SLANG})
    get_filename_component(REL_DIR ${REL_PATH} DIRECTORY)
    get_filename_component(BASE_NAME ${SLANG} NAME_WE)
    set(SPIRV "${CMAKE_BINARY_DIR}/shaders/${REL_DIR}/${BASE_NAME}.spv")
    compile_slang_shader(${SLANG} ${SPIRV} PROFILE glsl_460)
endforeach()

# Neural network compute kernels (shared across passes; live in src/shaders/neural/).
# Whitelist: other .slang files in this dir are import-only modules (adam/mlvec/mlp/hashgrid/sh).
set(NEURAL_KERNELS
    adam_kernel
    ema_kernel
    mlp_forward
    mlp_backward
    hashgrid_forward
    hashgrid_backward
    identity_forward
    sh_forward
    frequency_forward
    oneblob_forward
)
foreach(NK ${NEURAL_KERNELS})
    set(NK_SLANG "${SLANG_NEURAL_DIR}/${NK}.slang")
    set(NK_SPV   "${CMAKE_BINARY_DIR}/shaders/neural/${NK}.spv")
    compile_slang_shader(${NK_SLANG} ${NK_SPV} PROFILE glsl_460)
endforeach()

add_custom_target(shaders ALL DEPENDS ${SPIRV_BINARY_FILES})

# generate project_config.h with build-time absolute paths (e.g. SHADER_ROOT)
configure_file(
    ${CMAKE_SOURCE_DIR}/src/project_config.h.in
    ${CMAKE_BINARY_DIR}/generated/project_config.h
    @ONLY
)

# Auto-discover pass subdirectories. Each dir under src/passes/ gets:
#   - added to PASSES_FILES (all *.cpp inside)
#   - added to PASS_INCLUDE_DIRS (so pass headers are reachable by name)
# Top-level src/passes/*.cpp (e.g. pass_base.cpp) is picked up separately.
set(PASS_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/src/passes)
file(GLOB PASSES_TOP CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/passes/*.cpp")
set(PASSES_FILES ${PASSES_TOP})
file(GLOB PASS_SUBS LIST_DIRECTORIES true CONFIGURE_DEPENDS
     "${CMAKE_SOURCE_DIR}/src/passes/*")
foreach(sub ${PASS_SUBS})
    if(IS_DIRECTORY ${sub})
        list(APPEND PASS_INCLUDE_DIRS ${sub})
        file(GLOB sub_cpps CONFIGURE_DEPENDS "${sub}/*.cpp")
        list(APPEND PASSES_FILES ${sub_cpps})
    endif()
endforeach()

# core library
file(GLOB CORE_FILES CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/src/core/*.cpp"
    "${CMAKE_SOURCE_DIR}/src/neural/*.cpp"
    "${CMAKE_SOURCE_DIR}/src/scene/*.cpp"
    "${CMAKE_SOURCE_DIR}/src/gui/*.cpp"
)
add_library(core STATIC ${CORE_FILES})
target_include_directories(core PUBLIC
    ${CMAKE_BINARY_DIR}/generated
    ${CMAKE_SOURCE_DIR}/src/core
    ${CMAKE_SOURCE_DIR}/src/neural
    ${CMAKE_SOURCE_DIR}/src/scene
    ${CMAKE_SOURCE_DIR}/src/gui
    ${Stb_INCLUDE_DIR}
)
target_link_libraries(core PUBLIC
    glm::glm
    Vulkan::Vulkan
    glfw
    imgui::imgui
    assimp::assimp
)

# passes library
add_library(passes STATIC ${PASSES_FILES})
target_include_directories(passes PUBLIC ${PASS_INCLUDE_DIRS})
target_link_libraries(passes PUBLIC
    core
    nlohmann_json::nlohmann_json
)

# main executable
add_executable(main "${CMAKE_SOURCE_DIR}/src/main.cpp")
target_link_libraries(main PRIVATE
    core
    passes
)

# Force linker to include all symbols from passes lib (needed for self-registration)
if(MSVC)
    target_link_options(main PRIVATE "/WHOLEARCHIVE:passes")
else()
    target_link_libraries(main PRIVATE -Wl,--whole-archive passes -Wl,--no-whole-archive)
endif()

add_dependencies(main shaders)
