#pragma once

#include "geometry.cuh"
#include "owl/common/math/vec.h"

template <int Dim, class FieldValueType> struct DiffusionRayGenData {
    OptixTraversableHandle acc_structure;
    Mesh mesh;
    DeviceConfig<Dim> config;
    int device_index;
    int device_count;
    float dt;
    float diffusion_coefficient;

    bool use_initial_value_as_boundary_value;
    FieldValueType boundary_value; // only used when use_initial_value_as_boundary_value is true

    owl::vec_t<float, Dim> *evaluation_point_buf;
    float *winding_number_buf;
    FieldValueType *new_field_buf;
    utils::randState_t *random_state_buf;
    int num_evaluation_points;

    float *grid_winding_number_buf;
    FieldValueType *grid_field_buf;
    owl::vec_t<float, Dim> *grid_advection_velocity_buf;
};

template <int Dim>
inline __device__ owl::vec_t<float, Dim> backtrace(
    const owl::vec_t<float, Dim> &position, const owl::vec_t<float, Dim> *velocity_buf, const float dt,
    const DeviceConfig<Dim> &config
) {
    const auto get_vel = [&](const owl::vec_t<float, Dim> &pos) {
        return utils::sample_buffer_linear(
            utils::domain_point_to_idx(pos, config.grid_res, config.domain_size), velocity_buf, config.grid_res,
            utils::WrapMode::ClampToEdge
        );
    };

    switch (config.advection_mode) {
    case AdvectionMode::Euler: {
        const owl::vec_t<float, Dim> k1 = get_vel(position);
        return position - dt * k1;
    }
    case AdvectionMode::MacCormack:
    default: {
        const owl::vec_t<float, Dim> k1 = get_vel(position);
        const owl::vec_t<float, Dim> k2 = get_vel(position - dt * k1);
        return position - 0.5f * dt * (k1 + k2);
    }
    case AdvectionMode::RK3: {
        const owl::vec_t<float, Dim> k1 = get_vel(position);
        const owl::vec_t<float, Dim> k2 = get_vel(position - 0.5f * dt * k1);
        const owl::vec_t<float, Dim> k3 = get_vel(position - 0.75f * dt * k2);
        return position - dt * (2.f * k1 + 3.f * k2 + 4.f * k3) / 9.f;
    }
    case AdvectionMode::RK4: {
        const owl::vec_t<float, Dim> k1 = get_vel(position);
        const owl::vec_t<float, Dim> k2 = get_vel(position - 0.5f * dt * k1);
        const owl::vec_t<float, Dim> k3 = get_vel(position - 0.5f * dt * k2);
        const owl::vec_t<float, Dim> k4 = get_vel(position - dt * k3);
        return position - dt * (k1 + 2.f * k2 + 2.f * k3 + k4) / 6.f;
    }
    }
}

template <int Dim, typename T>
inline __device__ T get_advected_value(
    const owl::vec_t<float, Dim> &position, const owl::vec_t<float, Dim> *velocity_buf, const T *buffer, const float dt,
    const DeviceConfig<Dim> &config
) {
    owl::vec_t<float, Dim> idx =
        utils::domain_point_to_idx(backtrace(position, velocity_buf, dt, config), config.grid_res, config.domain_size);
    return utils::sample_buffer_linear(idx, buffer, config.grid_res, utils::WrapMode::ClampToEdge);
}
