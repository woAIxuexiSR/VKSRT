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
if(NOT SLANGC_EXECUTABLE)
    if(DEFINED ENV{VULKAN_SDK})
        set(SLANGC_EXECUTABLE "$ENV{VULKAN_SDK}/bin/slangc")
    else()
        message(FATAL_ERROR "slangc not found")
    endif()
endif()

file(GLOB_RECURSE SLANG_FILES "src/passes/**/*.slang")

set(SPIRV_BINARY_FILES "")
foreach(SLANG ${SLANG_FILES})
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

# core library
include_directories(
    ${CMAKE_SOURCE_DIR}/src/core
    ${CMAKE_SOURCE_DIR}/src/scene
    ${CMAKE_SOURCE_DIR}/src/gui
    ${CMAKE_SOURCE_DIR}/src/passes
    ${CMAKE_SOURCE_DIR}/src/passes/triangle
    ${CMAKE_SOURCE_DIR}/src/passes/ray_tracing
)
include_directories(${Stb_INCLUDE_DIR})

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

# main executable
add_executable(main "${CMAKE_SOURCE_DIR}/src/main.cpp")
target_link_libraries(main PRIVATE
    glm::glm
    Vulkan::Vulkan
    glfw
    imgui
    core
)
add_dependencies(main shaders)
