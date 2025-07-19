cmake_minimum_required(VERSION 3.19)

set(CMAKE_PROJECT_NAME BejzakEngine)

project(${CMAKE_PROJECT_NAME})

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_program(CLANG_TIDY_EXE NAMES "clang-tidy")

# if(CLANG_TIDY_EXE)
#     message(STATUS "Found clang-tidy: ${CLANG_TIDY_EXE}")
#     set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXE}")
# endif()

find_package(Vulkan REQUIRED)

if(NOT Vulkan_FOUND)
    message(FATAL_ERROR "Vulkan not found!")
endif()

# find_package(Boost REQUIRED COMPONENTS ${BOOST_COMPONENTS})
# 
# if(NOT Boost_FOUND)
#     message(WARNING "Boost not found!")
# else()
#     add_definitions(-DBOOST_ENABLED)
# 
#     set(Boost_USE_STATIC_LIBS ON)
#     set(Boost_USE_MULTITHREADED ON)
#     set(Boost_USE_STATIC_RUNTIME OFF)
# 
#     set(BOOST_COMPONENTS filesystem system)
# endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build (Debug or Release)" FORCE)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "Building in Release mode.")
else()
    message(STATUS "Building in Debug mode.")
    add_definitions(-DVALIDATION_LAYERS_ENABLED)
endif()

add_definitions(-DGLM_FORCE_NO_CTOR_INIT)

if(WIN32)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /W3")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
elseif(UNIX)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -Wall -Wextra -pedantic -O0")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")
endif()

add_subdirectory(bejzak_engine)

add_executable(${CMAKE_PROJECT_NAME} main.cpp application.cpp)
target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE LibStrongTypes)
target_link_libraries(${CMAKE_PROJECT_NAME} PUBLIC Status Window Instance Surface DebugMessenger PhysicalDevice MemoryAllocator LogicalDevice Renderpass Swapchain Buffer CommandBuffer Framebuffer Pipeline OBJLoader TinyGLTFLoader DescriptorSet Camera Screenshot Entity ComponentSystem ECSRegistry Object ThreadPool Primitives AssetManager)
target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE OBJLoader)

if(WIN32)
    set(SHADER_COMPILE_SCRIPT
    "${PROJECT_SOURCE_DIR}/bejzak_engine/shaders/compile_shaders.bat")
elseif(UNIX)
    set(SHADER_COMPILE_SCRIPT
    "${PROJECT_SOURCE_DIR}/bejzak_engine/shaders/compile_shaders.sh")
endif()

file(CHMOD ${SHADER_COMPILE_SCRIPT} PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ)

add_custom_target(
    CompileShaders
    COMMAND ${SHADER_COMPILE_SCRIPT}
    COMMENT "Compiling shaders...")

add_dependencies(${CMAKE_PROJECT_NAME} CompileShaders)

add_compile_definitions(SHADERS_PATH="${PROJECT_SOURCE_DIR}/bejzak_engine/shaders/")
add_compile_definitions(TEXTURES_PATH="${PROJECT_SOURCE_DIR}/assets/textures/")
add_compile_definitions(MODELS_PATH="${PROJECT_SOURCE_DIR}/assets/models/")