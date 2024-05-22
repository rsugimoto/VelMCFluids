/* The core implementation of Neumann WoB for velocity-based fluids. */
#pragma once

#include "fluids_common.cuh"

template <int Dim> struct ProjectionVPLRecord {
    owl::vec_t<float, Dim> position;
    float value;
};

template <int Dim> struct ProjectionRayGenData {
    OptixTraversableHandle acc_structure;
    Mesh mesh;
    Mesh pseudo_boundary_mesh;
    DeviceConfig<Dim> config;
    int device_index;
    int device_count;
    float dt;

    owl::vec_t<float, Dim> *evaluation_point_buf;
    float *winding_number_buf;
    owl::vec_t<float, Dim> *new_velocity_buf;
    utils::randState_t *random_state_buf;
    int num_evaluation_points;

    float *grid_winding_number_buf;
    owl::vec_t<float, Dim> *grid_velocity_buf;
    owl::vec_t<float, Dim> *grid_advection_velocity_buf;
    owl::vec_t<float, Dim> solid_velocity;
    owl::vec_t<float, Dim + 1> *velocity_source_buf;
    int velocity_source_count;
    ProjectionVPLRecord<Dim> *vpl_data_buf;
};
