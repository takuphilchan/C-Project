cmake_minimum_required(VERSION 3.10)
project(cppDL)

find_package(OpenCV REQUIRED)
find_package(Qt6 COMPONENTS Widgets Core Gui REQUIRED)

set(CMAKE_CXX_STANDARD 11)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Qt6Widgets_INCLUDE_DIRS} ${Qt6Core_INCLUDE_DIRS} ${Qt6Gui_INCLUDE_DIRS})

add_executable(main main.cpp)

target_link_libraries(main
    ${OpenCV_LIBS}
    Qt6::Widgets
    Qt6::Core
    Qt6::Gui
)
