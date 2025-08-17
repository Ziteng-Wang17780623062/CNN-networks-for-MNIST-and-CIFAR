# 数据结构课程设计 - [CNN开放题]
# 版权所有 (c) [2025] [爱玩的尼克]
# 根据 [MIT License] 许可证发布。详情请见项目根目录下的 LICENSE 文件。
# CMake 最低版本要求
cmake_minimum_required(VERSION 3.16)
# 启用 UTF-8 支持策略 (CMP0118)
cmake_policy(SET CMP0118 NEW)

# 项目定义
project(CNN VERSION 0.1 LANGUAGES CXX)

# 自动包含当前目录
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# 启用 Qt 自动处理工具
set(CMAKE_AUTOUIC ON)  # 自动处理 UI 文件
set(CMAKE_AUTOMOC ON)  # 自动处理元对象系统
set(CMAKE_AUTORCC ON)  # 自动处理资源文件

# C++ 标准设置
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)  # 必须使用 C++17

# 查找 Qt6 核心组件
find_package(Qt6 REQUIRED COMPONENTS Core Gui Charts)
# 兼容 Qt5/Qt6 的查找方式
find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Widgets)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Widgets)

# 用户可配置路径
set(USER_TORCH_DIR "" CACHE PATH "Libtorch 安装路径")
set(USER_OPENCV_DIR "" CACHE PATH "OpenCV 安装路径")

# LibTorch 配置
if(NOT "${USER_TORCH_DIR}" STREQUAL "")
    set(Torch_DIR ${USER_TORCH_DIR}/share/cmake/Torch)
endif()
find_package(Torch REQUIRED)

# OpenCV 配置
if(NOT "${USER_OPENCV_DIR}" STREQUAL "")
    set(OpenCV_DIR ${USER_OPENCV_DIR} CACHE PATH "OpenCV 库路径" FORCE)
endif()
find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc imgcodecs)

# 项目源文件定义
set(PROJECT_SOURCES
    main.cpp
    Mainwindow.cpp
    Mainwindow.h
    Mainwindow.ui

    # MNIST 相关文件
    model_mnist.h
    model_mnist.cpp
    testthread.h
    testthread.cpp
    trainingthread.h
    trainingthread.cpp
    handwritingdialog.h
    handwritingdialog.cpp

    # CIFAR 相关文件
    model_cifar.h
    model_cifar.cpp
    cifardataset.h
    cifardataset.cpp
    testthread_cifar.h
    testthread_cifar.cpp
    trainingthread_cifar.h
    trainingthread_cifar.cpp
)

# 创建可执行文件
if(${QT_VERSION_MAJOR} GREATER_EQUAL 6)
    qt_add_executable(CNN
        MANUAL_FINALIZATION
        ${PROJECT_SOURCES}
    )
else()
    add_executable(CNN ${PROJECT_SOURCES})
endif()

# 链接 Qt Widgets 模块
target_link_libraries(CNN PRIVATE 
    Qt${QT_VERSION_MAJOR}::Widgets
    Qt6::Core
    Qt6::Gui
    Qt6::Charts
)

# 链接 Torch 和 OpenCV 库
target_link_libraries(CNN PUBLIC "${TORCH_LIBRARIES}")
target_link_libraries(CNN PUBLIC ${OpenCV_LIBS})

# Qt6 的最终化处理
if(QT_VERSION_MAJOR EQUAL 6)
    qt_finalize_executable(CNN)
endif()

# 安装配置
include(GNUInstallDirs)
install(TARGETS CNN
    BUNDLE DESTINATION .
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# 平台特定配置
if(MSVC)
    # 设置 UTF-8 编码
    add_compile_options(
        /utf-8
        /source-charset:utf-8
        /execution-charset:utf-8
    )
    # 屏蔽 size_t 到 int 转换警告
    target_compile_options(CNN PRIVATE /wd4267)
endif()

# 跨平台设置
set_target_properties(CNN PROPERTIES
    WIN32_EXECUTABLE TRUE
    MACOSX_BUNDLE TRUE
    MACOSX_BUNDLE_GUI_IDENTIFIER com.example.CNN
    MACOSX_BUNDLE_BUNDLE_VERSION ${PROJECT_VERSION}
    MACOSX_BUNDLE_SHORT_VERSION_STRING ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
)