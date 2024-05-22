#include "common.cuh"
#include "fluids_common.cuh"
#include "geometry.cuh"
#include "utils.hpp"

// draws a sample from gamma(1, 1) distribution.
inline __device__ float gamma_1_sample(utils::randState_t &rand_state) {
    constexpr float eps = std::numeric_limits<float>::epsilon();
    return -log(std::max(1.0f - utils::rand_uniform(rand_state), eps));
}

// draws a sample from gamma(1.5, 1) distribution.
inline __device__ float gamma_1_5_sample(utils::randState_t &rand_state) {
    float normal_sample = utils::rand_normal(rand_state);
    return gamma_1_sample(rand_state) + normal_sample * normal_sample / 2.f;
}

// WoB for diffusion equation.
// It is assumed that the boundary condition is Dirichlet stored in the initial condition grid.
// We could add a few more parameters to change this behaviour.
template <int Dim, class FieldValueType>
__device__ FieldValueType diffuse(
    const owl::vec_t<float, Dim> &original_position, utils::randState_t &rand_state,
    const float *grid_winding_number_buf, const FieldValueType *grid_field_buf,
    const owl::vec_t<float, Dim> *grid_advection_velocity_buf, const bool use_initial_value_as_boundary_value,
    const FieldValueType &boundary_value, const float dt, const float diffusion_coefficient,
    const OptixTraversableHandle acc_structure, const Mesh &mesh, const DeviceConfig<Dim> &config
) {
    const auto get_field_value = [&](const owl::vec_t<float, Dim> &x) -> FieldValueType {
        if (grid_advection_velocity_buf)
            return get_advected_value(x, grid_advection_velocity_buf, grid_field_buf, dt, config);
        else {
            owl::vec_t<float, Dim> idx = utils::domain_point_to_idx(x, config.grid_res, config.domain_size);
            return utils::sample_buffer_linear(idx, grid_field_buf, config.grid_res, utils::WrapMode::ClampToEdge);
        }
    };

    // We assume that the boundary condition is Dirichlet and it does not depend on time.
    const auto get_boundary_value = [&](const owl::vec_t<float, Dim> &p) -> FieldValueType {
        if (use_initial_value_as_boundary_value)
            return get_field_value(p);
        else
            return boundary_value;
    };

    const auto initial_condition_term = [&](const owl::vec_t<float, Dim> &p, const float t,
                                            const int num_samples) -> FieldValueType {
        utils::KahanSum<FieldValueType> sum;
        for (int v = 0; v < num_samples; v++) {
            const float gamma = Dim == 3 ? gamma_1_5_sample(rand_state) : gamma_1_sample(rand_state);
            const owl::vec_t<float, Dim> dir = uniform_direction_sample<Dim>(rand_state);
            const owl::vec_t<float, Dim> z = p + 2.f * sqrtf(t * gamma) * dir;
            float multiplier;
            if (config.num_winding_samples == 0) {
                owl::vec_t<float, Dim> idx = utils::domain_point_to_idx(z, config.grid_res, config.domain_size);
                float winding_number = utils::sample_buffer_linear(
                    idx, grid_winding_number_buf, config.grid_res, utils::WrapMode::ClampToBorder
                );
                multiplier = config.domain_type == DomainType::BoundedDomain ? winding_number : 1.0f + winding_number;
            } else
                multiplier = volume_integral_multiplier(
                    acc_structure, z, rand_state, config.num_winding_samples, mesh.num_primitives, config.domain_type
                ); // inlining this gives an incorrect result.
            sum += multiplier * get_field_value(z);
        }
        return sum.sum / (float)num_samples;
    };

    const float current_t = dt * diffusion_coefficient; // trace back to time zero from here.

    FieldValueType result = initial_condition_term(original_position, current_t, config.num_volume_samples_direct);

    if (mesh.num_primitives > 0) {
        utils::KahanSum<FieldValueType> sample_sum;
        for (int s = 0; s < config.num_path_samples; s++) {
            FieldValueType path_contribution = utils::zero<FieldValueType>();
            float t = current_t;
            float weight = -1.f;

            // primitive id = -1 indicates that it is actually not a point on boundary.
            BoundaryPoint<Dim> y_prev{original_position, -1};
            while (true) {
                const auto [y, num_intersections] =
                    line_intersection_boundary_sample<Dim>(acc_structure, y_prev, rand_state);

                const float gamma = Dim == 3 ? gamma_1_5_sample(rand_state) : gamma_1_sample(rand_state);
                t = t - owl::length2(y.position - y_prev.position) / (4.f * gamma);
                // if t is not finite, that must mean t is negative without numerical computation, too.
                if (t < 0.f || !std::isfinite(t) || num_intersections == 0) break;

                const int intersection_sign =
                    dot(owl::cast_dim<Dim>(owl::xfmNormal(mesh.transform, mesh.normal_buf[y.prim_id])),
                        y.position - y_prev.position) > 0
                        ? 1
                        : -1;
                weight *= -intersection_sign * num_intersections;
                path_contribution +=
                    weight * (get_boundary_value(y.position) -
                              initial_condition_term(y.position, t, config.num_volume_samples_indirect));
                y_prev = y;
            }
            sample_sum += path_contribution;
        }
        result += sample_sum.sum / (float)config.num_path_samples;
    }
    return result;
}

OPTIX_RAYGEN_PROGRAM(diffusionScalar2DRayGen)() {
    const DiffusionRayGenData<2, float> &self = owl::getProgramData<DiffusionRayGenData<2, float>>();
    const owl::vec3i idx_3d = optixGetLaunchIndex();

    // round robin scheduling with 32 x indices per chunk.
    const int responsible_device = (idx_3d.x >> 5) % self.device_count;
    if (self.device_index != responsible_device) return;

    const owl::vec3i dim_3d = optixGetLaunchDimensions();
    const int idx = utils::flatten(idx_3d, dim_3d);
    if (idx >= self.num_evaluation_points) return;

    const owl::vec2f position = self.evaluation_point_buf[idx];
    utils::randState_t rand_state = self.random_state_buf[idx];

    float new_field_value = 0.0f;
    float blend_weight = self.config.domain_type == DomainType::BoundedDomain ? self.winding_number_buf[idx]
                                                                              : 1.f + self.winding_number_buf[idx];

    if (blend_weight > 0.0f)
        new_field_value += blend_weight * diffuse(
                                              position, rand_state, self.grid_winding_number_buf, self.grid_field_buf,
                                              self.grid_advection_velocity_buf,
                                              self.use_initial_value_as_boundary_value, self.boundary_value, self.dt,
                                              self.diffusion_coefficient, self.acc_structure, self.mesh, self.config
                                          );

    self.new_field_buf[idx] = new_field_value;
    self.random_state_buf[idx] = rand_state;
}

OPTIX_RAYGEN_PROGRAM(diffusionScalar3DRayGen)() {
    const DiffusionRayGenData<3, float> &self = owl::getProgramData<DiffusionRayGenData<3, float>>();
    const owl::vec3i idx_3d = optixGetLaunchIndex();

    // round robin scheduling with 32 x indices per chunk.
    const int responsible_device = (idx_3d.x >> 5) % self.device_count;
    if (self.device_index != responsible_device) return;

    const owl::vec3i dim_3d = optixGetLaunchDimensions();
    const int idx = utils::flatten(idx_3d, dim_3d);
    if (idx >= self.num_evaluation_points) return;

    const owl::vec3f position = self.evaluation_point_buf[idx];
    utils::randState_t rand_state = self.random_state_buf[idx];

    float new_field_value = 0.0f;
    float blend_weight = self.config.domain_type == DomainType::BoundedDomain ? self.winding_number_buf[idx]
                                                                              : 1.f + self.winding_number_buf[idx];

    if (blend_weight > 0.0f)
        new_field_value += blend_weight * diffuse(
                                              position, rand_state, self.grid_winding_number_buf, self.grid_field_buf,
                                              self.grid_advection_velocity_buf,
                                              self.use_initial_value_as_boundary_value, self.boundary_value, self.dt,
                                              self.diffusion_coefficient, self.acc_structure, self.mesh, self.config
                                          );

    self.new_field_buf[idx] = new_field_value;
    self.random_state_buf[idx] = rand_state;
}

OPTIX_RAYGEN_PROGRAM(diffusionVector2DRayGen)() {
    const DiffusionRayGenData<2, owl::vec2f> &self = owl::getProgramData<DiffusionRayGenData<2, owl::vec2f>>();
    const owl::vec3i idx_3d = optixGetLaunchIndex();

    // round robin scheduling with 32 x indices per chunk.
    const int responsible_device = (idx_3d.x >> 5) % self.device_count;
    if (self.device_index != responsible_device) return;

    const owl::vec3i dim_3d = optixGetLaunchDimensions();
    const int idx = utils::flatten(idx_3d, dim_3d);
    if (idx >= self.num_evaluation_points) return;

    const owl::vec2f position = self.evaluation_point_buf[idx];
    utils::randState_t rand_state = self.random_state_buf[idx];

    owl::vec2f new_field_value = utils::zero<owl::vec2f>();
    float blend_weight = self.config.domain_type == DomainType::BoundedDomain ? self.winding_number_buf[idx]
                                                                              : 1.f + self.winding_number_buf[idx];

    if (blend_weight > 0.0f)
        new_field_value += blend_weight * diffuse(
                                              position, rand_state, self.grid_winding_number_buf, self.grid_field_buf,
                                              self.grid_advection_velocity_buf,
                                              self.use_initial_value_as_boundary_value, self.boundary_value, self.dt,
                                              self.diffusion_coefficient, self.acc_structure, self.mesh, self.config
                                          );
    self.new_field_buf[idx] = new_field_value;
    self.random_state_buf[idx] = rand_state;
}

OPTIX_RAYGEN_PROGRAM(diffusionVectorRayGen3D)() {
    const DiffusionRayGenData<3, owl::vec3f> &self = owl::getProgramData<DiffusionRayGenData<3, owl::vec3f>>();
    const owl::vec3i idx_3d = optixGetLaunchIndex();

    // round robin scheduling with 32 x indices per chunk.
    const int responsible_device = (idx_3d.x >> 5) % self.device_count;
    if (self.device_index != responsible_device) return;

    const owl::vec3i dim_3d = optixGetLaunchDimensions();
    const int idx = utils::flatten(idx_3d, dim_3d);
    if (idx >= self.num_evaluation_points) return;

    const owl::vec3f position = self.evaluation_point_buf[idx];
    utils::randState_t rand_state = self.random_state_buf[idx];

    owl::vec3f new_field_value = utils::zero<owl::vec3f>();
    float blend_weight = self.config.domain_type == DomainType::BoundedDomain ? self.winding_number_buf[idx]
                                                                              : 1.f + self.winding_number_buf[idx];

    if (blend_weight > 0.0f)
        new_field_value += blend_weight * diffuse(
                                              position, rand_state, self.grid_winding_number_buf, self.grid_field_buf,
                                              self.grid_advection_velocity_buf,
                                              self.use_initial_value_as_boundary_value, self.boundary_value, self.dt,
                                              self.diffusion_coefficient, self.acc_structure, self.mesh, self.config
                                          );

    self.new_field_buf[idx] = new_field_value;
    self.random_state_buf[idx] = rand_state;
}