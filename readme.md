# VelMCFluids
This is a reference OptiX implementation of our SIGGRAPH North America 2024 paper, ["Velocity-Based Monte Carlo Fluids"](https://https://rsugimoto.net/VelMCFluidsProject/), Sugimoto, Batty, and Hachisuka. 2024.
This repository contains the simulation program for all examples shown in the paper. Note this repository does not contain all the visualization scripts to produce the figures and does not contain other methods (e.g., Baty et al. 2007 or Rioux-Lavoie and Sugimoto et al. 2022) discussed in the paper.

## Directory Structure
- `common`: contains the implementations of the core functionalities, such as the projection and diffusion steps of the solver.
- `apps`: contains .cu files with the `main` function. For each of the .cu files, an executable will be compiled. Each of the files contained in this directory utilizes the routines defined in the `common` directory.
    - `velocity_fluids.cu`: most basic 2D advection-projection solver.
    - `velocity_fluids_divfree_advection.cu`: 2D advection-projection solver with pointwise divergence-free advection.
    - `velocity_fluids_flip.cu`: 2D advection-projection solver with FLIP/PIC advection.
    - `velocity_fluids_reflection.cu`: 2D advection-reflection solver.
    - `velocity_fluids_3d.cu`: most basic 3D advection-projection solver.
    - `projection_test.cu`: simplified code used to measure the error of the projection step.
- `configs`: contains scene configuration JSON files.
- `objs`: contains Wavefront OBJ files used as boundary shape.

## Dependencies and Acknowledgement
All dependencies are included as git submodules or in the `commmon` directory. When cloning the repository, use the following:

        git clone --recurse-submodules  https://github.com/rsugimoto/VelMCFluids

This repository utilizes the following external libraries.
- [OWL](https://github.com/owl-project/owl): an OptiX 7 wrapper library.
- [nlohmann/json](https://github.com/nlohmann/json): a header-only C++ JSON parser.
- [tinyobjloader](https://github.com/tinyobjloader/tinyobjloader): a header-only Wavefront OBJ loader.

## Requirement
The program compiles and runs on x86-64 Linux machines with NVIDIA OptiX 7/8 support.
The program is tested with OptiX SDK versions 7.7.0 and 8.0.0.

## Compile and Run
To compile the program, you need to have the NVIDIA OptiX SDK installed.
Download and install it from https://developer.nvidia.com/designworks/optix/download.
The code is tested with SDK versions 7.7.0 and 8.0.0.
Set environment variable `OptiX_INSTALL_DIR` to the SDK directory.

Once OptiX SDK is installed, you can run the standard cmake routine to compile the program:

    mkdir build
    cd build
    cmake -DCMAKE_BUILD_TYPE=Release ..
    make -j8

This will make several executable files in the `build` directory. The compilation may take a few minutes. You can run programs from the `build` directory with a JSON scene configuration file, for example:

    ./velocity_fluids_div_free_advection ../configs/config_cohomology.json

In each of the scene configuration JSON files, the `binary` entry specifies which executable file should be used to get the result. The program generates outputs under the directory specified in the JSON file (e.g. `results/results_cohomology/raw`), in a binary format.

Once you get the outputs, you can visualize the result in your favorite way. For the "cohomology" example scene, we attached a visualization Python script:

    python visualize.py results/results_cohomology

The resulting PNG files will be saved in `results/results_cohomology/png`.

