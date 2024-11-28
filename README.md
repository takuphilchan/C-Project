# Deep learning
This repository contains code snippets and implementations of various deep learning concepts, algorithms, and techniques. It serves as a collection of practical examples to help understand and explore key principles in deep learning.

# Installations
This guide will walk you through the process of setting up a development environment for C++ with OpenCV and Qt using MSYS2, as well as configuring Visual Studio Code (VSCode) for development. By the end of this guide, you will have all the necessary tools installed and configured to start working on your C++ projects with OpenCV and Qt on Windows.

---

## Step 1: Clone the Repository

1. **Clone the Repository:**
   - First, clone this repository to your local machine. Open a terminal (such as Command Prompt or Git Bash) and run the following command:
     ```
     git clone https://github.com/takuphilchan/Deep-Learning.git
     ```
   
2. **Navigate to the Project Folder:**
   - Once the repository is cloned, navigate to the project folder:
     ```
     cd "Deep Learning"
     ```

---

## Step 2: Install Visual Studio Code (VSCode)

1. **Download VSCode:**
   - Visit the [Visual Studio Code website](https://code.visualstudio.com/) and download the latest stable version of VSCode for Windows.

2. **Install VSCode:**
   - Run the installer and follow the installation instructions. Make sure to select the options to add VSCode to your PATH and install necessary dependencies during the installation process.

3. **Install Extensions for C++ Development:**
   - Open VSCode and go to the Extensions view (click on the Extensions icon in the Activity Bar on the side of the window).
   - Install the following extensions for C++ development:
     - **C/C++** by Microsoft (for IntelliSense, debugging, and code navigation)
     - **CMake Tools** (if you plan to use CMake for your project)
     - **CodeLLDB** (for debugging)

4. **Configure VSCode for MSYS2:**
   - In VSCode, go to `File > Preferences > Settings` and search for `C++`. Make sure the compiler path is set to the `g++` executable from MSYS2. For example:
     - `"C_Cpp.default.compilerPath": "C:\\msys64\\mingw64\\bin\\g++.exe"`

---

## Step 3: Install MSYS2 and Set Up Development Environment

1. **Download and Install MSYS2:**
   - Visit the [MSYS2 website](https://www.msys2.org/) and download the appropriate version for your system (32-bit or 64-bit).
   - Run the installer and follow the installation instructions. It’s recommended to install MSYS2 to the default directory.

2. **Update MSYS2:**
   - Open the MSYS2 terminal (either MSYS2 MinGW 32-bit or MSYS2 MinGW 64-bit, depending on your system) and run the following commands to update the package database and core system packages:
     ```
     pacman -Syu
     ```
   - If prompted to restart the terminal, close and reopen it, and then run the command again to ensure everything is up-to-date.

---

## Step 4: Install C++ Compiler and Dependencies

1. **Install the C++ Compiler:**
   - In the MSYS2 terminal, install the `gcc` package (which includes the C++ compiler) by running:
     ```
     pacman -S mingw-w64-x86_64-gcc
     ```

2. **Install OpenCV:**
   - To install OpenCV and its dependencies, run the following command:
     ```
     pacman -S mingw-w64-x86_64-opencv
     ```

3. **Install Qt (Optional for GUI Development):**
   - If you want to develop graphical applications using Qt, install Qt by running:
     ```
     pacman -S mingw-w64-x86_64-qt5
     ```
   - For the latest version of Qt (Qt6), run:
     ```
     pacman -S mingw-w64-x86_64-qt6
     ```

---

## Step 5: Configure the Project in VSCode

1. **Open the Project Folder in VSCode:**
   - Open VSCode and navigate to your project folder by selecting `File > Open Folder` and choosing the folder where you cloned your repository.

2. **Configure Build System:**
   - If you're using a simple `g++` compiler setup, you can configure the build tasks by creating a `.vscode/tasks.json` file. If you're using CMake, create a `CMakeLists.txt` file and use the CMake extension in VSCode.
   - Set up the build task for compiling C++ code with OpenCV and Qt. For example, the command for compiling with `g++` should include paths for OpenCV and Qt libraries (set up based on where they are installed).

3. **Configure Debugging:**
   - Set up debugging in VSCode by creating a `.vscode/launch.json` file. This will allow you to run and debug your program directly from VSCode.
   - Make sure the `miDebuggerPath` points to the correct `gdb` executable from MSYS2, and ensure the compiler and libraries are correctly linked.

---

## Step 6: Building and Running Your Project

1. **Build the Project:**
   - In the VSCode terminal, navigate to your project directory if you haven’t already, and run the build task:
     ```
     Ctrl+Shift+B
     ```
   - This will compile your project using the configured build system.

2. **Run the Project:**
   - After building the project, you can run it directly from the VSCode terminal by executing:
     ```
     ./your_program_name
     ```

---

## Step 7: Troubleshooting

- **Compiler Not Found:**
  If the compiler is not found, make sure that MSYS2 is installed correctly and that the `mingw64` path is included in the system's `PATH` environment variable.
  
- **Missing Libraries or Headers:**
  Ensure that OpenCV and Qt are installed and configured correctly in the `tasks.json` or `CMakeLists.txt` file. You may need to include additional paths to OpenCV and Qt libraries.

- **VSCode Debugger Issues:**
  Ensure that `gdb` and `g++` are correctly set up in the MSYS2 environment. Check the paths in your `launch.json` file to make sure everything points to the right executables.

---
For additional resources, check out the following documentation:

- [OpenCV Documentation](https://docs.opencv.org/)
- [Qt Documentation](https://doc.qt.io/)
- [MSYS2 Documentation](https://www.msys2.org/docs/)
- [VSCode Documentation](https://code.visualstudio.com/docs/)


