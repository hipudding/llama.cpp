if (NOT SOC_TYPE)
    set (SOC_TYPE "Ascend910B3")
endif()

file(GLOB SRC_FILES
    dequantize_q4_0.cpp
    quantize_q4_0.cpp
)

string(TOLOWER "${CMAKE_BUILD_TYPE}" lowercase_CMAKE_BUILD_TYPE)
if(${lowercase_CMAKE_BUILD_TYPE} STREQUAL "debug")
    if (NOT DEFINED ENV{CMAKE_PREFIX_PATH})
    set(CMAKE_PREFIX_PATH ${CANN_INSTALL_DIR}/tools/tikicpulib/lib/cmake)
    endif()

    find_package(tikicpulib REQUIRED)
    add_library(cann_kernels ${SRC_FILES} ascendc_kernels.cpp)
    target_link_libraries(cann_kernels PRIVATE
        ascendcl
        tikicpulib::ascend910B1
    )

    target_compile_features(cann_kernels PRIVATE cxx_std_17)

else()
    string(TOLOWER ${SOC_TYPE} SOC_VERSION)
    set(ASCEND_CANN_PACKAGE_PATH ${CANN_INSTALL_DIR})
    set(RUN_MODE "npu" CACHE STRING "run mode: npu/sim")

    if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
        set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
    elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/ascendc_devkit/tikcpp/samples/cmake)
        set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/ascendc_devkit/tikcpp/samples/cmake)
    else()
        message(FATAL_ERROR "ascendc_kernel_cmake does not exist, please check whether the compiler package is installed.")
    endif()
    include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

    ascendc_library(ascendc_kernels STATIC
        ${SRC_FILES}
    )
    #ascendc_compile_definitions(ascendc_kernels PRIVATE -DASCENDC_DUMP)

    add_library(cann_kernels STATIC ascendc_kernels.cpp)
    target_link_libraries(cann_kernels PUBLIC ascendc_kernels)
endif()