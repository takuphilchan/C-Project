# Makefile
CXX = g++
CXXFLAGS = -I"C:/msys64/mingw64/include/qt6" \
            -I"C:/msys64/mingw64/include/qt6/QtWidgets" \
            -I"C:/msys64/mingw64/include/opencv4" \
            -I"C:/msys64/mingw64/include/opencv4/opencv2" \
            -std=c++11

LDFLAGS = -L"C:/msys64/mingw64/lib" \
          -lQt6Widgets \
          -lQt6Core \
          -lQt6Gui \
          -lopencv_core \
          -lopencv_imgcodecs \
          -lopencv_highgui \
          -lopencv_imgproc

TARGET = convolutional_filters.exe
SOURCES = convolutional_filters.cpp
OBJECTS = $(SOURCES:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(OBJECTS) -o $(TARGET) $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET)
