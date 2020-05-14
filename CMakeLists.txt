cmake_minimum_required(VERSION 3.15)
project(vulkan_perceptron)

set(CMAKE_CXX_STANDARD 17)

FILE(GLOB sources src/*.cpp)

add_executable(vulkan_perceptron main.cpp ${sources})

target_include_directories(vulkan_perceptron PUBLIC includes)

find_package(Vulkan REQUIRED)

target_include_directories(${PROJECT_NAME} PRIVATE ${Vulkan_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} Vulkan::Vulkan)

find_package(PkgConfig REQUIRED)
pkg_search_module(GLM REQUIRED glm)
include_directories(${GLM_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${GLM_LIBRARY_DIRS})

find_package(glfw3 REQUIRED)
include_directories(${GLFW_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${GLFW_LIBRARIES} glfw)

include_directories(shaders)