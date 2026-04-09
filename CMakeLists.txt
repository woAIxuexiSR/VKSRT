cmake_minimum_required(VERSION 3.10)

project(VKSRT LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Debug)

# dependencies via vcpkg
find_package(glm CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(Vulkan REQUIRED)
find_package(Stb REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

# imgui
set(IMGUI_DIR "D:/software/imgui/")
include_directories(${IMGUI_DIR})
add_library(imgui STATIC
    "${IMGUI_DIR}/imgui.cpp"
    "${IMGUI_DIR}/imgui_draw.cpp"
    "${IMGUI_DIR}/imgui_widgets.cpp"
    "${IMGUI_DIR}/imgui_tables.cpp"
    "${IMGUI_DIR}/imgui_demo.cpp"
    "${IMGUI_DIR}/backends/imgui_impl_glfw.cpp"
    "${IMGUI_DIR}/backends/imgui_impl_vulkan.cpp"
)
target_link_libraries(imgui PRIVATE glfw Vulkan::Vulkan)

# compile slang shaders
set(SPIRV_BINARY_FILES "")
if(NOT SLANGC_EXECUTABLE)
    if(DEFINED ENV{VULKAN_SDK})
        set(SLANGC_EXECUTABLE "$ENV{VULKAN_SDK}/bin/slangc")
    else()
        message(FATAL_ERROR "slangc not found")
    endif()
endif()

# RT slang shader: single file with all entry points compiled to one SPV
set(RT_SLANG "${CMAKE_SOURCE_DIR}/src/passes/ray_tracing/ray_tracing.slang")
set(RT_SLANG_DIR "${CMAKE_SOURCE_DIR}/src/passes/ray_tracing")
set(RT_OUT_DIR "${CMAKE_BINARY_DIR}/shaders/ray_tracing")
set(RT_SPV "${RT_OUT_DIR}/ray_tracing.spv")

add_custom_command(
    OUTPUT ${RT_SPV}
    COMMAND ${CMAKE_COMMAND} -E make_directory "${RT_OUT_DIR}"
    COMMAND ${SLANGC_EXECUTABLE} ${RT_SLANG} -I${RT_SLANG_DIR} -target spirv -entry raygenMain -entry missMain -entry closestHitMain -o ${RT_SPV}
    DEPENDS ${RT_SLANG}
    COMMENT "Compiling ray_tracing.slang (all entries) to single SPIR-V"
)
list(APPEND SPIRV_BINARY_FILES ${RT_SPV})

# compile other slang shaders (single entry point each)
file(GLOB_RECURSE SLANG_FILES "src/passes/**/*.slang")

foreach(SLANG ${SLANG_FILES})
    # Skip RT slang file (handled above with multi-entry compilation)
    get_filename_component(SLANG_NAME ${SLANG} NAME)
    if(SLANG_NAME STREQUAL "ray_tracing.slang")
        continue()
    endif()
    get_filename_component(FILE_NAME ${SLANG} NAME)
    get_filename_component(SLANG_DIR ${SLANG} DIRECTORY)

    # Get relative path from src/passes for output directory structure
    file(RELATIVE_PATH REL_PATH ${CMAKE_SOURCE_DIR}/src/passes ${SLANG})
    get_filename_component(DIR_PATH ${REL_PATH} PATH)
    set(SPIRV "${CMAKE_BINARY_DIR}/shaders/${DIR_PATH}/${FILE_NAME}.spv")

    add_custom_command(
        OUTPUT ${SPIRV}
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/shaders/${DIR_PATH}"
        COMMAND ${SLANGC_EXECUTABLE} ${SLANG} -I${SLANG_DIR} -profile glsl_460 -target spirv -o ${SPIRV}
        DEPENDS ${SLANG}
        COMMENT "Compiling ${FILE_NAME} to SPIR-V (slang)"
    )

    list(APPEND SPIRV_BINARY_FILES ${SPIRV})
endforeach()

add_custom_target(shaders ALL DEPENDS ${SPIRV_BINARY_FILES})

# include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/src/core
    ${CMAKE_SOURCE_DIR}/src/scene
    ${CMAKE_SOURCE_DIR}/src/gui
    ${CMAKE_SOURCE_DIR}/src/passes
    ${CMAKE_SOURCE_DIR}/src/passes/ray_tracing
    ${CMAKE_SOURCE_DIR}/src/passes/tonemap
    ${CMAKE_SOURCE_DIR}/src/passes/blit
)
include_directories(${Stb_INCLUDE_DIR})

# core library
file(GLOB CORE_FILES
    "${CMAKE_SOURCE_DIR}/src/core/*.cpp"
    "${CMAKE_SOURCE_DIR}/src/scene/*.cpp"
    "${CMAKE_SOURCE_DIR}/src/gui/*.cpp"
)
add_library(core STATIC ${CORE_FILES})
target_link_libraries(core PRIVATE
    glm::glm
    Vulkan::Vulkan
    glfw
    imgui
)

# passes library
file(GLOB PASSES_FILES
    "${CMAKE_SOURCE_DIR}/src/passes/*.cpp"
    "${CMAKE_SOURCE_DIR}/src/passes/ray_tracing/*.cpp"
    "${CMAKE_SOURCE_DIR}/src/passes/tonemap/*.cpp"
    "${CMAKE_SOURCE_DIR}/src/passes/blit/*.cpp"
)
add_library(passes STATIC ${PASSES_FILES})
target_link_libraries(passes PRIVATE
    glm::glm
    Vulkan::Vulkan
    glfw
    imgui
    core
    nlohmann_json::nlohmann_json
)

# main executable
add_executable(main "${CMAKE_SOURCE_DIR}/src/main.cpp")
target_link_libraries(main PRIVATE
    glm::glm
    Vulkan::Vulkan
    glfw
    imgui
    core
    passes
    nlohmann_json::nlohmann_json
)

# Force linker to include all symbols from passes lib (needed for self-registration)
if(MSVC)
    target_link_options(main PRIVATE "/WHOLEARCHIVE:passes")
else()
    target_link_libraries(main PRIVATE -Wl,--whole-archive passes -Wl,--no-whole-archive)
endif()

add_dependencies(main shaders)
