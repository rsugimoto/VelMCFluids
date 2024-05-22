#pragma once

#include "velocity_fluids.cuh"
#include <json.hpp>
#include <owl/owl_host.h>
#include <string>
#include <thrust/universal_vector.h>

// DeviceConfig<Dim> struct is used in OptiX programs.
// Confing extends DeviceConfig<Dim> with additional fields used on the host side.
template <int Dim> struct Config : public DeviceConfig<Dim> {
    nlohmann::json json_config; // expose this to allow custom programs to access the json data
    int time_steps;
    int num_evaluation_points;
    owl::vec_t<float, Dim> solid_velocity;
    std::string output_dir;
    std::string obj_file;
    bool invert_normals;
    float dt;
    float velocity_diffusion_coefficient;
    float concentration_diffusion_coefficient;
    float temperature_diffusion_coefficient;
    float buoyancy_alpha;
    float buoyancy_beta;
    float concentration_rate;
    float temperature_rate;
    owl::vec_t<float, Dim> buoyancy_gravity;
    bool enable_inplace_advection;
    bool enable_vpl;
    owl::vec_t<float, Dim> obstacle_shift;
    float obstacle_scale;
    thrust::universal_vector<owl::vec_t<float, Dim + 1>> velocity_sources;
    int scene_number;
};

template <int Dim> Config<Dim> load_config(const char *file);

template <int Dim> class Scene {
  public:
    Scene(const Config<Dim> &config);

    void initialize_evaluation_points(thrust::universal_vector<owl::vec_t<float, Dim>> &evaluation_point);

    void winding_number(
        const thrust::universal_vector<owl::vec_t<float, Dim>> &evaluation_point,
        thrust::universal_vector<float> &winding_number_buf, thrust::universal_vector<utils::randState_t> &random_state
    );

    void project(
        const thrust::universal_vector<owl::vec_t<float, Dim>> &evaluation_point,
        const thrust::universal_vector<float> &winding_number_buf,
        thrust::universal_vector<owl::vec_t<float, Dim>> &new_velocity,
        thrust::universal_vector<utils::randState_t> &random_state,

        const thrust::universal_vector<float> &grid_winding_number_buf,
        const thrust::universal_vector<owl::vec_t<float, Dim>> &grid_velocity,
        const thrust::universal_vector<owl::vec_t<float, Dim>> &grid_advection_velocity,

        thrust::universal_vector<ProjectionVPLRecord<Dim>> &vpl_data,
        thrust::universal_vector<utils::randState_t> &vpl_random_state,

        const float dt, const bool enable_inplace_advection,

        const thrust::universal_vector<owl::vec_t<float, Dim + 1>> &velocity_sources =
            thrust::universal_vector<owl::vec_t<float, Dim + 1>>()
    );

    void project_pointwise(
        const thrust::universal_vector<owl::vec_t<float, Dim>> &evaluation_point,
        const thrust::universal_vector<float> &winding_number_buf,
        thrust::universal_vector<owl::vec_t<float, Dim>> &new_velocity,
        thrust::universal_vector<utils::randState_t> &random_state,

        const thrust::universal_vector<float> &grid_winding_number_buf,
        const thrust::universal_vector<owl::vec_t<float, Dim>> &grid_velocity,
        const thrust::universal_vector<owl::vec_t<float, Dim>> &grid_advection_velocity,

        const float dt, const bool enable_inplace_advection,

        const thrust::universal_vector<owl::vec_t<float, Dim + 1>> &velocity_sources =
            thrust::universal_vector<owl::vec_t<float, Dim + 1>>()
    );

    void project_vpl_construct(
        thrust::universal_vector<ProjectionVPLRecord<Dim>> &vpl_data,
        thrust::universal_vector<utils::randState_t> &vpl_random_state,

        const thrust::universal_vector<float> &grid_winding_number_buf,
        const thrust::universal_vector<owl::vec_t<float, Dim>> &grid_velocity,
        const thrust::universal_vector<owl::vec_t<float, Dim>> &grid_advection_velocity,

        const float dt, const bool enable_inplace_advection,

        const thrust::universal_vector<owl::vec_t<float, Dim + 1>> &velocity_sources =
            thrust::universal_vector<owl::vec_t<float, Dim + 1>>()
    );

    void project_vpl_gather(
        const thrust::universal_vector<owl::vec_t<float, Dim>> &evaluation_point,
        const thrust::universal_vector<float> &winding_number_buf,
        thrust::universal_vector<owl::vec_t<float, Dim>> &new_velocity,
        thrust::universal_vector<utils::randState_t> &random_state,

        const thrust::universal_vector<float> &grid_winding_number_buf,
        const thrust::universal_vector<owl::vec_t<float, Dim>> &grid_velocity,
        const thrust::universal_vector<owl::vec_t<float, Dim>> &grid_advection_velocity,
        const thrust::universal_vector<ProjectionVPLRecord<Dim>> &vpl_data,

        const float dt, const bool enable_inplace_advection,

        const thrust::universal_vector<owl::vec_t<float, Dim + 1>> &velocity_sources =
            thrust::universal_vector<owl::vec_t<float, Dim + 1>>()
    );

    void project_vpl(
        const thrust::universal_vector<owl::vec_t<float, Dim>> &evaluation_point,
        const thrust::universal_vector<float> &winding_number_buf,
        thrust::universal_vector<owl::vec_t<float, Dim>> &new_velocity,
        thrust::universal_vector<utils::randState_t> &random_state,

        const thrust::universal_vector<float> &grid_winding_number_buf,
        const thrust::universal_vector<owl::vec_t<float, Dim>> &grid_velocity,
        const thrust::universal_vector<owl::vec_t<float, Dim>> &grid_advection_velocity,

        thrust::universal_vector<ProjectionVPLRecord<Dim>> &vpl_data,
        thrust::universal_vector<utils::randState_t> &vpl_random_state,

        const float dt, const bool enable_inplace_advection,

        const thrust::universal_vector<owl::vec_t<float, Dim + 1>> &velocity_sources =
            thrust::universal_vector<owl::vec_t<float, Dim + 1>>()
    );

    void diffuse_velocity(
        const thrust::universal_vector<owl::vec_t<float, Dim>> &evaluation_point,
        const thrust::universal_vector<float> &winding_number_buf,
        thrust::universal_vector<owl::vec_t<float, Dim>> &new_velocity,
        thrust::universal_vector<utils::randState_t> &random_state,

        const thrust::universal_vector<float> &grid_winding_number_buf,
        const thrust::universal_vector<owl::vec_t<float, Dim>> &grid_velocity,
        const thrust::universal_vector<owl::vec_t<float, Dim>> &grid_advection_velocity,

        float diffusion_coefficient, const float dt, const bool enable_inplace_advection
    );

    void diffuse_scalar(
        const thrust::universal_vector<owl::vec_t<float, Dim>> &evaluation_point,
        const thrust::universal_vector<float> &winding_number_buf, thrust::universal_vector<float> &new_field,
        thrust::universal_vector<utils::randState_t> &random_state,

        const thrust::universal_vector<float> &grid_winding_number_buf,
        const thrust::universal_vector<float> &grid_field,
        const thrust::universal_vector<owl::vec_t<float, Dim>> &grid_advection_velocity,

        float diffusion_coefficient, const float dt, const bool enable_inplace_advection
    );

    template <typename FieldValueType>
    void advect(
        const thrust::universal_vector<owl::vec_t<float, Dim>> &evaluation_point,
        const thrust::universal_vector<owl::vec_t<float, Dim>> &velocity,
        const thrust::universal_vector<FieldValueType> &field, thrust::universal_vector<FieldValueType> &new_field,
        const float dt
    );

    template <typename FieldValueType>
    void particle_to_grid(
        const thrust::universal_vector<owl::vec_t<float, Dim>> &particle_positions,
        const thrust::universal_vector<FieldValueType> &particle_values,
        thrust::universal_vector<FieldValueType> &grid_field, thrust::universal_vector<float> &grid_weight_sum,
        int order = 2
    );

    template <typename FieldValueType>
    void grid_breadth_first_fill(
        const thrust::universal_vector<float> &grid_winding_number_buf,
        thrust::universal_vector<FieldValueType> &grid_field
    );

    void setTranslation(const owl::vec_t<float, Dim> &translation);

  private:
    const Config<Dim> &config;
    OWLContext context{0};
    OWLGroup world{0};
    OWLRayGen projection_ray_gen{0};
    OWLRayGen projection_vpl_construct_ray_gen{0};
    OWLRayGen projection_vpl_gather_ray_gen{0};
    OWLRayGen vector_diffusion_ray_gen{0};
    OWLRayGen scalar_diffusion_ray_gen{0};
    OWLRayGen winding_number_ray_gen{0};
    owl::affine3f transform;
};
