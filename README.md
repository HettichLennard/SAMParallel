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
- *cnf_file_name (line 1.297):* Specifies the path to the .cnf file that describes the feature model and the file name itself.
- *sampled_variants_size (line 1.300):* Represents the hyper-parameter *k*, i.e. the number of variant proposals generated with ContextSAT.
- *post_optimization_samples (line 1.875):* Represents the hyper-parameter *epoch*, i.e. the number of revisited and optimized older variants in the sample during each optimization phase.
- *optimization_epochs (line 1.877):* Represents the hyper-parameter *optimIter*, i.e. the max. number of flips per optimization call.
- *max_retries_per_optimization (line 1.878)* and *min_retries_per_optimization (line 1.879):* Represents the upper and lower bounds of the hyper-parameter *searchBudget*. The value of *searchBudget* takes the upper bound at the beginning and decays towards the lower bound during sampling.
- *parallel_tasks (line 1.886):* Depending on the used GPU, the targeted number of parallel GPU tasks may has to be lowered to ensure optimal performance!

## Sampling Results
The caclulated sample is stored in the arrays *current_sample* (on CPU) and *current_sample_c* (on GPU). The integer *current_sample_size*, respectively *current_sample_size_c* stores the sample size.
