cmake_minimum_required(VERSION 3.24)
# Set this before enabling CUDA: nvcc's default may be sm_75, while the
# compressors use __reduce_add_sync, which requires compute capability 8.0+.
# Preserve explicit cache/parent-project settings and the CUDAARCHS environment.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND "$ENV{CUDAARCHS}" STREQUAL "")
  set(CMAKE_CUDA_ARCHITECTURES 80 CACHE STRING "CUDA architectures (80 or newer)")
endif()
project(FPcompress VERSION 1.0.0 LANGUAGES CXX CUDA)

set(fpcompress_architectures_to_check "${CMAKE_CUDA_ARCHITECTURES}")
if(CMAKE_CUDA_ARCHITECTURES STREQUAL "native")
  set(fpcompress_architectures_to_check "${CMAKE_CUDA_ARCHITECTURES_NATIVE}")
elseif(CMAKE_CUDA_ARCHITECTURES MATCHES "^all(-major)?$" OR NOT CMAKE_CUDA_ARCHITECTURES)
  message(FATAL_ERROR
    "FPcompress requires CUDA architectures 80 or newer for __reduce_add_sync. "
    "Use an explicit list, e.g. -DCMAKE_CUDA_ARCHITECTURES=80.")
endif()
foreach(architecture IN LISTS fpcompress_architectures_to_check)
  if(architecture MATCHES "^([0-9]+)[af]?(-real|-virtual)?$" AND CMAKE_MATCH_1 LESS 80)
    message(FATAL_ERROR
      "FPcompress requires CUDA architectures 80 or newer for __reduce_add_sync; "
      "selected '${architecture}'. Reconfigure with -DCMAKE_CUDA_ARCHITECTURES=80 "
      "or the architecture of your deployment GPU.")
  endif()
endforeach()
unset(fpcompress_architectures_to_check)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
find_package(CUDAToolkit REQUIRED)

set(sources src/fpcompress.cu)
foreach(precision IN ITEMS single double)
  foreach(mode IN ITEMS speed ratio)
    foreach(operation IN ITEMS compressor decompressor)
      list(APPEND sources ${precision}_src/${mode}-${operation}-${precision}.cu)
    endforeach()
  endforeach()
endforeach()
add_library(fpcompress STATIC ${sources})
add_library(FPcompress::fpcompress ALIAS fpcompress)
# A convenient alternate name for consumers that name codec backends explicitly.
add_library(fpcompress_cuda INTERFACE)
target_link_libraries(fpcompress_cuda INTERFACE fpcompress)
add_library(FPcompress::cuda ALIAS fpcompress_cuda)
set_target_properties(fpcompress_cuda PROPERTIES EXPORT_NAME cuda)
set_target_properties(fpcompress PROPERTIES POSITION_INDEPENDENT_CODE ON
  CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
target_compile_features(fpcompress PUBLIC cxx_std_17)
target_compile_definitions(fpcompress PRIVATE FPCOMPRESS_LIBRARY)
target_include_directories(fpcompress PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(fpcompress PUBLIC CUDA::cudart)
target_compile_options(fpcompress PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)

install(TARGETS fpcompress fpcompress_cuda EXPORT FPcompressTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES include/fpcompress/fpcompress.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/fpcompress)
install(FILES LICENSE DESTINATION ${CMAKE_INSTALL_DATADIR}/FPcompress)
configure_package_config_file(cmake/FPcompressConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/FPcompressConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/FPcompress)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/FPcompressConfigVersion.cmake
  VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
install(EXPORT FPcompressTargets NAMESPACE FPcompress::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/FPcompress)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/FPcompressConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/FPcompressConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/FPcompress)
