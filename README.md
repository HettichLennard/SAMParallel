# SAMParallel
Implementation of the SAMParallel t-wise Interaction Sampling Approach in Visual Studio IDE 2022

## Requirements
- A PC with a Windows 11 operating system.
- A Nvidia GPU, we suggest at least 4GB of VRAM to execute SAMParallel for larger Software Product Lines (SPL).
- An installation of the Nvidia CUDA Toolkit (https://developer.nvidia.com/cuda-downloads), we used Version 12.8 to test SAMParallel.
- An installation of the Visual Studio IDE 2022 (https://visualstudio.microsoft.com/de/vs/), the installation has to be done **after** the previous step!

## Installation Steps
- **Create VS 2022 Project:** Open Visual Studio IDE 2022, create a new project and select the CUDA Runtime from the project templates.
- **Copy Code Files from Repository:** Copy all files with the ending .cc, .h, and .cu under CUDA_GPU_Test_3/CUDA_GPU_Test_3 to the equivalent path of your repository.
- **Add Code Files from Repository:** After opening the created VS 2022 project, add all previously copied files to the project via the Project Explorer (Add > Existing Element).
- **Run SAMParallel:** Open the file *kernel.cu* in VS 2022, compile, and run it. This file contains the SAMParallel implementation and will execute the sampling process.

## Parameter Settings

## Sampling Results
