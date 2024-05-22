cmake_minimum_required(VERSION 3.1)

project(main LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}") 

set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${OptiX_INSTALL_DIR})
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -mtune=native")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # for clangd
set(CMAKE_COLOR_DIAGNOSTICS ON)

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
cmake_policy(SET CMP0104 OLD) # for CUDA_ARCHITECTURES
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda --expt-relaxed-constexpr -std=c++17")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG -Wall")

set(OWL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/owl)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${OWL_DIR}/owl/cmake/")
add_subdirectory(${OWL_DIR} EXCLUDE_FROM_ALL)

embed_ptx(
  OUTPUT_TARGET
    common-ptx
  PTX_LINK_LIBRARIES
    owl::owl
  SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/common/fluids_common_optix.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/common/velocity_fluids_optix.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/common/geometry_optix.cu
)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/common)

set (
    APP_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/apps/velocity_fluids.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/apps/velocity_fluids_reflection.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/apps/velocity_fluids_divfree_advection.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/apps/velocity_fluids_flip.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/apps/velocity_fluids_3d.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/apps/projection_test.cu
)
foreach( APP_SOURCE ${APP_SOURCES} )
    file(RELATIVE_PATH EXECUTABLE_NAME ${CMAKE_CURRENT_SOURCE_DIR} ${APP_SOURCE})
    string( REPLACE "apps/" "" EXECUTABLE_NAME ${EXECUTABLE_NAME} )
    string( REPLACE ".cu" "" EXECUTABLE_NAME ${EXECUTABLE_NAME} )

    add_executable( ${EXECUTABLE_NAME} ${APP_SOURCE} ${CMAKE_CURRENT_SOURCE_DIR}/common/velocity_fluids_host.cu)
    target_link_libraries(${EXECUTABLE_NAME}
        PRIVATE
            common-ptx
            owl::owl
    )
endforeach()