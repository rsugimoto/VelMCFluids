/* The core implementation of Neumann WoB for velocity-based fluids. */
#include "common.cuh"
#include "velocity_fluids.cuh"
#include <cmath>

inline __device__ float
max_domain_corner_distance(const owl::vec2f &position, const owl::vec_t<float, 2> &domain_size) {
    float sampling_radius2 =
        max(max(owl::length2(position - owl::vec2f(-domain_size.x / 2, -domain_size.y / 2)),
                owl::length2(position - owl::vec2f(-domain_size.x / 2, +domain_size.y / 2))),
            max(owl::length2(position - owl::vec2f(+domain_size.x / 2, -domain_size.y / 2)),
                owl::length2(position - owl::vec2f(+domain_size.x / 2, +domain_size.y / 2))));
    return sqrtf(sampling_radius2);
}

inline __device__ float
max_domain_corner_distance(const owl::vec3f &position, const owl::vec_t<float, 3> &domain_size) {
    float sampling_radius2 =
        max(max(max(owl::length2(position - owl::vec3f(-domain_size.x / 2, -domain_size.y / 2, -domain_size.z / 2)),
                    owl::length2(position - owl::vec3f(-domain_size.x / 2, -domain_size.y / 2, +domain_size.z / 2))),
                max(owl::length2(position - owl::vec3f(-domain_size.x / 2, +domain_size.y / 2, -domain_size.z / 2)),
                    owl::length2(position - owl::vec3f(-domain_size.x / 2, +domain_size.y / 2, +domain_size.z / 2)))),
            max(max(owl::length2(position - owl::vec3f(+domain_size.x / 2, -domain_size.y / 2, -domain_size.z / 2)),
                    owl::length2(position - owl::vec3f(+domain_size.x / 2, -domain_size.y / 2, +domain_size.z / 2))),
                max(owl::length2(position - owl::vec3f(+domain_size.x / 2, +domain_size.y / 2, -domain_size.z / 2)),
                    owl::length2(position - owl::vec3f(+domain_size.x / 2, +domain_size.y / 2, +domain_size.z / 2)))));
    return sqrtf(sampling_radius2);
}

template <int Dim>
__device__ owl::vec_t<float, Dim> project(
    const owl::vec_t<float, Dim> &original_position, utils::randState_t &rand_state,
    const float *grid_winding_number_buf, const owl::vec_t<float, Dim> *grid_velocity_buf,
    const owl::vec_t<float, Dim> *grid_advection_velocity_buf, const owl::vec_t<float, Dim> &solid_velocity,
    const owl::vec_t<float, Dim + 1> *velocity_source_buf, const int velocity_source_count, const float dt,
    const OptixTraversableHandle acc_structure, const Mesh &mesh, const Mesh &pseudo_boundary_mesh,
    const DeviceConfig<Dim> &config
) {
    const owl::vec_t<float, Dim> half_domain_size = config.domain_size / 2.f;

    const auto get_velocity = [&](const owl::vec_t<float, Dim> &x) -> owl::vec_t<float, Dim> {
        if (grid_advection_velocity_buf)
            return get_advected_value(x, grid_advection_velocity_buf, grid_velocity_buf, dt, config);
        else {
            owl::vec_t<float, Dim> idx = utils::domain_point_to_idx(x, config.grid_res, config.domain_size);
            return utils::sample_buffer_linear(idx, grid_velocity_buf, config.grid_res, utils::WrapMode::ClampToEdge);
        }
    };

    const auto is_point_outside_solid = [&](const owl::vec_t<float, Dim> &y) -> float {
        float multiplier;
        if (config.num_winding_samples == 0) {
            owl::vec_t<float, Dim> idx = utils::domain_point_to_idx(y, config.grid_res, config.domain_size);
            float winding_number = utils::sample_buffer_linear(
                idx, grid_winding_number_buf, config.grid_res, utils::WrapMode::ClampToBorder
            );
            multiplier = config.domain_type == DomainType::BoundedDomain ? winding_number : 1.f + winding_number;
        } else
            multiplier = volume_integral_multiplier(
                acc_structure, y, rand_state, config.num_winding_samples, mesh.num_primitives, config.domain_type
            );
        return multiplier;
    };

    const auto is_point_inside_simulation_box = [&](const owl::vec_t<float, Dim> &y) -> int {
        if constexpr (Dim == 2)
            return (abs(y.x) <= half_domain_size.x && abs(y.y) <= half_domain_size.y);
        else
            return (abs(y.x) <= half_domain_size.x && abs(y.y) <= half_domain_size.y && abs(y.z) <= half_domain_size.z);
    };

    const auto estimate_volume_term = [&](const owl::vec_t<float, Dim> &x, const owl::vec_t<float, Dim> &x_vel,
                                          const int num_samples) -> owl::vec_t<float, Dim> {
        float sampling_radius = max_domain_corner_distance(x, config.domain_size);
        utils::KahanSum<owl::vec_t<float, Dim>> sum;
        for (int v = 0; v < num_samples; v++) {
            auto [r_vec, inv_pdf] = strongly_singular_ball_sample<Dim>(sampling_radius, rand_state);
            const float divisor = Dim == 3 ? owl::length2(r_vec) * owl::length(r_vec) : owl::length2(r_vec);
            owl::vec_t<float, Dim> r_hat = owl::normalize(r_vec);

            if (config.enable_antithetic_sampling) v += 1; // count the second sample.
            for (int i = 0; i < (int)config.enable_antithetic_sampling + 1; i++) {
                if (i == 1) {
                    r_vec = -r_vec;
                    r_hat = -r_hat;
                }

                const owl::vec_t<float, Dim> y = x + r_vec;
                const float multiplier = is_point_outside_solid(y) * is_point_inside_simulation_box(y);
                const owl::vec_t<float, Dim> vel_diff = get_velocity(y) - x_vel;
                sum += multiplier * inv_pdf / divisor * (Dim * dot(r_hat, vel_diff) * r_hat - vel_diff);
            }
        }
        return sum.sum / (num_samples * (2 * (Dim - 1) * M_PIf32));
    };

    const auto dGdx = [&](const owl::vec_t<float, Dim> &r_vec) -> owl::vec_t<float, Dim> {
        float divisor = Dim == 2 ? owl::length(r_vec) : owl::length2(r_vec);
        divisor = max(divisor, config.dGdx_regularization);
        owl::vec_t<float, Dim> result = owl::normalize(r_vec) / (divisor * (2 * (Dim - 1) * M_PIf32));
        return owl::isfinite(result) ? result : utils::zero<owl::vec_t<float, Dim>>();
    };

    const auto estimate_pseudo_boundary_term = [&](const owl::vec_t<float, Dim> &x, const owl::vec_t<float, Dim> &x_vel,
                                                   const int num_samples) -> owl::vec_t<float, Dim> {
        if (config.domain_type == DomainType::BoundedDomain) return utils::zero<owl::vec_t<float, Dim>>();
        utils::KahanSum<owl::vec_t<float, Dim>> sum;
        for (int s = 0; s < num_samples; s++) {
            auto [point, inv_pdf] = cdf_boundary_sample<Dim>(pseudo_boundary_mesh, rand_state);
            inv_pdf *= is_point_outside_solid(point.position);
            sum += dGdx(point.position - x) *
                   (inv_pdf * dot(owl::cast_dim<Dim>(owl::xfmNormal(
                                      pseudo_boundary_mesh.transform, pseudo_boundary_mesh.normal_buf[point.prim_id]
                                  )),
                                  get_velocity(point.position) - x_vel));
        }
        return sum.sum / (float)num_samples;
    };

    const auto estimate_velocity_source_term = [&](const owl::vec_t<float, Dim> &x) -> owl::vec_t<float, Dim> {
        utils::KahanSum<owl::vec_t<float, Dim>> sum;
        for (int s = 0; s < velocity_source_count; s++) {
            const owl::vec_t<float, Dim> source_position = owl::cast_dim<Dim>(velocity_source_buf[s]);
            sum += velocity_source_buf[s][Dim] * dGdx(x - source_position);
        }
        return sum.sum;
    };

    owl::vec_t<float, Dim> original_vel = get_velocity(original_position);

    owl::vec_t<float, Dim> pressure_grad_total =
        -(estimate_volume_term(original_position, original_vel, config.num_volume_samples_direct) +
          estimate_pseudo_boundary_term(original_position, original_vel, config.num_pseudo_boundary_samples_direct) +
          estimate_velocity_source_term(original_position));

    if (mesh.num_primitives > 0) {
        utils::KahanSum<owl::vec_t<float, Dim>> pressure_grad_sum;
        for (int s = 0; s < config.num_path_samples; s++) {
            // length 0 contribution is computed outside this loop.
            owl::vec_t<float, Dim> pressure_grad = utils::zero<owl::vec_t<float, Dim>>();

            auto [source_point, source_inv_pdf] = cdf_boundary_sample<Dim>(mesh, rand_state);
            source_inv_pdf *= is_point_inside_simulation_box(source_point.position);
            const owl::vec_t<float, Dim> source_normal =
                owl::cast_dim<Dim>(owl::xfmNormal(mesh.transform, mesh.normal_buf[source_point.prim_id]));
            const owl::vec_t<float, Dim> source_vel = get_velocity(source_point.position);
            const owl::vec_t<float, Dim> source_terms =
                estimate_volume_term(source_point.position, source_vel, config.num_volume_samples_indirect) +
                estimate_pseudo_boundary_term(
                    source_point.position, source_vel, config.num_pseudo_boundary_samples_indirect
                ) +
                estimate_velocity_source_term(source_point.position);

            // length 0.5 and 1 contributions.
            pressure_grad += dGdx(source_point.position - original_position) *
                             (source_inv_pdf *
                              dot(source_normal, original_vel + source_vel + 2.f * (source_terms - solid_velocity)));

            auto [point_x, num_intersections_x] =
                line_intersection_boundary_sample<Dim>(acc_structure, source_point, rand_state);
            num_intersections_x *= is_point_inside_simulation_box(point_x.position);
            const owl::vec_t<float, Dim> normal_x =
                owl::cast_dim<Dim>(owl::xfmNormal(mesh.transform, mesh.normal_buf[point_x.prim_id]));
            const int intersection_sign_x = dot(normal_x, point_x.position - source_point.position) > 0 ? 1 : -1;
            const int si_x = intersection_sign_x * num_intersections_x;
            const owl::vec_t<float, Dim> vel_x = get_velocity(point_x.position);

            // for length n + 0.5 contributions
            float multiplier_n_5 = source_inv_pdf * si_x * dot(source_normal, source_vel - vel_x);
            // for length n + 1  contributions
            float multiplier_n =
                source_inv_pdf * si_x * 2.f * dot(source_normal, source_terms + source_vel - solid_velocity);

            // length 1.5 and 2 contribution
            pressure_grad += (multiplier_n_5 + multiplier_n) * dGdx(point_x.position - original_position);

            BoundaryPoint<Dim> point_y = point_x;
            for (int i = 2; i <= config.path_length; i++) {
                auto [point_x, num_intersections_x] =
                    line_intersection_boundary_sample<Dim>(acc_structure, point_y, rand_state);
                num_intersections_x *= is_point_inside_simulation_box(point_x.position);
                const owl::vec_t<float, Dim> normal_x =
                    owl::cast_dim<Dim>(owl::xfmNormal(mesh.transform, mesh.normal_buf[point_x.prim_id]));
                const int intersection_sign_x = dot(normal_x, point_x.position - point_y.position) > 0 ? 1 : -1;
                const int si_x = intersection_sign_x * num_intersections_x;

                multiplier_n_5 *= si_x;
                multiplier_n *= si_x;
                if (i == config.path_length - 1) {
                    multiplier_n *= 0.5f;
                } else if (i == config.path_length) {
                    multiplier_n_5 *= 0.5f;
                    multiplier_n = 0.f;
                }

                pressure_grad += (multiplier_n_5 + multiplier_n) * dGdx(point_x.position - original_position);

                point_y = point_x;
            }

            pressure_grad_sum += pressure_grad;
        }
        pressure_grad_total += pressure_grad_sum.sum / (float)config.num_path_samples;
    }
    return original_vel - pressure_grad_total;
}

template <int Dim>
__device__ void project_vpl_construct(
    ProjectionVPLRecord<Dim> *vpl_data_buf, utils::randState_t &rand_state, const float *grid_winding_number_buf,
    const owl::vec_t<float, Dim> *grid_velocity_buf, const owl::vec_t<float, Dim> *grid_advection_velocity_buf,
    const owl::vec_t<float, Dim> &solid_velocity, const owl::vec_t<float, Dim + 1> *velocity_source_buf,
    const int velocity_source_count, const float dt, const OptixTraversableHandle acc_structure, const Mesh &mesh,
    const Mesh &pseudo_boundary_mesh, const DeviceConfig<Dim> &config
) {
    const owl::vec_t<float, Dim> half_domain_size = config.domain_size / 2.f;

    const auto get_velocity = [&](const owl::vec_t<float, Dim> &x) -> owl::vec_t<float, Dim> {
        if (grid_advection_velocity_buf)
            return get_advected_value(x, grid_advection_velocity_buf, grid_velocity_buf, dt, config);
        else {
            owl::vec_t<float, Dim> idx = utils::domain_point_to_idx(x, config.grid_res, config.domain_size);
            return utils::sample_buffer_linear(idx, grid_velocity_buf, config.grid_res, utils::WrapMode::ClampToEdge);
        }
    };

    const auto is_point_outside_solid = [&](const owl::vec_t<float, Dim> &y) -> float {
        float multiplier;
        if (config.num_winding_samples == 0) {
            owl::vec_t<float, Dim> idx = utils::domain_point_to_idx(y, config.grid_res, config.domain_size);
            float winding_number = utils::sample_buffer_linear(
                idx, grid_winding_number_buf, config.grid_res, utils::WrapMode::ClampToBorder
            );
            multiplier = config.domain_type == DomainType::BoundedDomain ? winding_number : 1.f + winding_number;
        } else
            multiplier = volume_integral_multiplier(
                acc_structure, y, rand_state, config.num_winding_samples, mesh.num_primitives, config.domain_type
            );
        return multiplier;
    };

    const auto is_point_inside_simulation_box = [&](const owl::vec_t<float, Dim> &y) -> float {
        if constexpr (Dim == 2)
            return (abs(y.x) <= half_domain_size.x && abs(y.y) <= half_domain_size.y);
        else
            return (abs(y.x) <= half_domain_size.x && abs(y.y) <= half_domain_size.y && abs(y.z) <= half_domain_size.z);
    };

    const auto estimate_volume_term = [&](const owl::vec_t<float, Dim> &x,
                                          const owl::vec_t<float, Dim> &x_vel) -> owl::vec_t<float, Dim> {
        float sampling_radius = max_domain_corner_distance(x, config.domain_size);
        utils::KahanSum<owl::vec_t<float, Dim>> sum;
        for (int v = 0; v < config.num_volume_samples_indirect; v++) {
            auto [r_vec, inv_pdf] = strongly_singular_ball_sample<Dim>(sampling_radius, rand_state);
            const float divisor = Dim == 3 ? owl::length2(r_vec) * owl::length(r_vec) : owl::length2(r_vec);
            owl::vec_t<float, Dim> r_hat = owl::normalize(r_vec);

            if (config.enable_antithetic_sampling) v += 1; // count the second sample.
            for (int i = 0; i < (int)config.enable_antithetic_sampling + 1; i++) {
                if (i == 1) {
                    r_vec = -r_vec;
                    r_hat = -r_hat;
                }

                const owl::vec_t<float, Dim> y = x + r_vec;
                const float multiplier = is_point_outside_solid(y) * is_point_inside_simulation_box(y);
                const owl::vec_t<float, Dim> vel_diff = get_velocity(y) - x_vel;
                sum += multiplier * inv_pdf / divisor * (Dim * dot(r_hat, vel_diff) * r_hat - vel_diff);
            }
        }
        return sum.sum / (config.num_volume_samples_indirect * (2 * (Dim - 1) * M_PIf32));
    };

    const auto dGdx = [&](const owl::vec_t<float, Dim> &r_vec) -> owl::vec_t<float, Dim> {
        float divisor = Dim == 2 ? owl::length(r_vec) : owl::length2(r_vec);
        divisor = max(divisor, config.dGdx_regularization);
        owl::vec_t<float, Dim> result = owl::normalize(r_vec) / (divisor * (2 * (Dim - 1) * M_PIf32));
        return owl::isfinite(result) ? result : utils::zero<owl::vec_t<float, Dim>>();
    };

    const auto estimate_pseudo_boundary_term = [&](const owl::vec_t<float, Dim> &x,
                                                   const owl::vec_t<float, Dim> &x_vel) -> owl::vec_t<float, Dim> {
        if (config.domain_type == DomainType::BoundedDomain) return utils::zero<owl::vec_t<float, Dim>>();
        utils::KahanSum<owl::vec_t<float, Dim>> sum;
        for (int s = 0; s < config.num_pseudo_boundary_samples_indirect; s++) {
            auto [point, inv_pdf] = cdf_boundary_sample<Dim>(pseudo_boundary_mesh, rand_state);
            inv_pdf *= is_point_outside_solid(point.position);
            sum += dGdx(point.position - x) *
                   (inv_pdf * dot(owl::cast_dim<Dim>(owl::xfmNormal(
                                      pseudo_boundary_mesh.transform, pseudo_boundary_mesh.normal_buf[point.prim_id]
                                  )),
                                  get_velocity(point.position) - x_vel));
        }
        return sum.sum / (float)config.num_pseudo_boundary_samples_indirect;
    };

    const auto estimate_velocity_source_term = [&](const owl::vec_t<float, Dim> &x) -> owl::vec_t<float, Dim> {
        utils::KahanSum<owl::vec_t<float, Dim>> sum;
        for (int s = 0; s < velocity_source_count; s++) {
            const owl::vec_t<float, Dim> source_position = owl::cast_dim<Dim>(velocity_source_buf[s]);
            sum += velocity_source_buf[s][Dim] * dGdx(x - source_position);
        }
        return sum.sum;
    };

    if (mesh.num_primitives > 0) {
        auto [source_point, source_inv_pdf] = cdf_boundary_sample<Dim>(mesh, rand_state);
        source_inv_pdf *= is_point_inside_simulation_box(source_point.position);
        const owl::vec_t<float, Dim> source_normal =
            owl::cast_dim<Dim>(owl::xfmNormal(mesh.transform, mesh.normal_buf[source_point.prim_id]));
        const owl::vec_t<float, Dim> source_vel = get_velocity(source_point.position);
        const owl::vec_t<float, Dim> source_terms = estimate_volume_term(source_point.position, source_vel) +
                                                    estimate_pseudo_boundary_term(source_point.position, source_vel) +
                                                    estimate_velocity_source_term(source_point.position);

        // length 0.5 and 1 contributions.
        vpl_data_buf[0] = {
            source_point.position,
            source_inv_pdf * dot(source_normal, source_vel + 2.f * (source_terms - solid_velocity))};

        auto [point_x, num_intersections_x] =
            line_intersection_boundary_sample<Dim>(acc_structure, source_point, rand_state);
        num_intersections_x *= is_point_inside_simulation_box(point_x.position);
        const owl::vec_t<float, Dim> normal_x =
            owl::cast_dim<Dim>(owl::xfmNormal(mesh.transform, mesh.normal_buf[point_x.prim_id]));
        const int intersection_sign_x = dot(normal_x, point_x.position - source_point.position) > 0 ? 1 : -1;
        const int si_x = intersection_sign_x * num_intersections_x;
        const owl::vec_t<float, Dim> vel_x = get_velocity(point_x.position);

        // for length n + 0.5 contributions
        float multiplier_n_5 = source_inv_pdf * si_x * dot(source_normal, source_vel - vel_x);
        // for length n + 1  contributions
        float multiplier_n =
            source_inv_pdf * si_x * 2.f * dot(source_normal, source_terms + source_vel - solid_velocity);

        // length 1.5 and 2 contribution
        vpl_data_buf[1] = {point_x.position, multiplier_n_5 + multiplier_n};

        BoundaryPoint<Dim> point_y = point_x;
        for (int i = 2; i <= config.path_length; i++) {
            auto [point_x, num_intersections_x] =
                line_intersection_boundary_sample<Dim>(acc_structure, point_y, rand_state);
            num_intersections_x *= is_point_inside_simulation_box(point_x.position);
            const owl::vec_t<float, Dim> normal_x =
                owl::cast_dim<Dim>(owl::xfmNormal(mesh.transform, mesh.normal_buf[point_x.prim_id]));
            const int intersection_sign_x = dot(normal_x, point_x.position - point_y.position) > 0 ? 1 : -1;
            const int si_x = intersection_sign_x * num_intersections_x;

            multiplier_n_5 *= si_x;
            multiplier_n *= si_x;
            if (i == config.path_length - 1) {
                multiplier_n *= 0.5f;
            } else if (i == config.path_length) {
                multiplier_n_5 *= 0.5f;
                multiplier_n = 0.f;
            }

            vpl_data_buf[i] = {point_x.position, multiplier_n_5 + multiplier_n};

            point_y = point_x;
        }
    }
}

template <int Dim>
__device__ owl::vec_t<float, Dim> project_vpl_gather(
    const owl::vec_t<float, Dim> &original_position, utils::randState_t &rand_state,
    const float *grid_winding_number_buf, const owl::vec_t<float, Dim> *grid_velocity_buf,
    const owl::vec_t<float, Dim> *grid_advection_velocity_buf, const owl::vec_t<float, Dim> &solid_velocity,
    const owl::vec_t<float, Dim + 1> *velocity_source_buf, const int velocity_source_count,
    const ProjectionVPLRecord<Dim> *vpl_data_buf, const float dt, const OptixTraversableHandle acc_structure,
    const Mesh &mesh, const Mesh &pseudo_boundary_mesh, const DeviceConfig<Dim> &config
) {
    const owl::vec_t<float, Dim> half_domain_size = config.domain_size / 2.f;

    const auto get_velocity = [&](const owl::vec_t<float, Dim> &x) -> owl::vec_t<float, Dim> {
        if (grid_advection_velocity_buf)
            return get_advected_value(x, grid_advection_velocity_buf, grid_velocity_buf, dt, config);
        else {
            owl::vec_t<float, Dim> idx = utils::domain_point_to_idx(x, config.grid_res, config.domain_size);
            return utils::sample_buffer_linear(idx, grid_velocity_buf, config.grid_res, utils::WrapMode::ClampToEdge);
        }
    };

    const auto is_point_outside_solid = [&](const owl::vec_t<float, Dim> &y) -> float {
        float multiplier;
        if (config.num_winding_samples == 0) {
            owl::vec_t<float, Dim> idx = utils::domain_point_to_idx(y, config.grid_res, config.domain_size);
            float winding_number = utils::sample_buffer_linear(
                idx, grid_winding_number_buf, config.grid_res, utils::WrapMode::ClampToBorder
            );
            multiplier = config.domain_type == DomainType::BoundedDomain ? winding_number : 1.f + winding_number;
        } else
            multiplier = volume_integral_multiplier(
                acc_structure, y, rand_state, config.num_winding_samples, mesh.num_primitives, config.domain_type
            );
        return multiplier;
    };

    const auto is_point_inside_simulation_box = [&](const owl::vec_t<float, Dim> &y) -> float {
        if constexpr (Dim == 2)
            return (abs(y.x) <= half_domain_size.x && abs(y.y) <= half_domain_size.y);
        else
            return (abs(y.x) <= half_domain_size.x && abs(y.y) <= half_domain_size.y && abs(y.z) <= half_domain_size.z);
    };

    const auto estimate_volume_term = [&](const owl::vec_t<float, Dim> &x,
                                          const owl::vec_t<float, Dim> &x_vel) -> owl::vec_t<float, Dim> {
        float sampling_radius = max_domain_corner_distance(x, config.domain_size);
        utils::KahanSum<owl::vec_t<float, Dim>> sum;
        for (int v = 0; v < config.num_volume_samples_direct; v++) {
            auto [r_vec, inv_pdf] = strongly_singular_ball_sample<Dim>(sampling_radius, rand_state);
            const float divisor = Dim == 3 ? owl::length2(r_vec) * owl::length(r_vec) : owl::length2(r_vec);
            owl::vec_t<float, Dim> r_hat = owl::normalize(r_vec);

            if (config.enable_antithetic_sampling) v += 1; // count the second sample.
            for (int i = 0; i < (int)config.enable_antithetic_sampling + 1; i++) {
                if (i == 1) {
                    r_vec = -r_vec;
                    r_hat = -r_hat;
                }

                const owl::vec_t<float, Dim> y = x + r_vec;
                const float multiplier = is_point_outside_solid(y) * is_point_inside_simulation_box(y);
                const owl::vec_t<float, Dim> vel_diff = get_velocity(y) - x_vel;
                sum += multiplier * inv_pdf / divisor * (Dim * dot(r_hat, vel_diff) * r_hat - vel_diff);
            }
        }
        return sum.sum / (config.num_volume_samples_direct * (2 * (Dim - 1) * M_PIf32));
    };

    const auto dGdx = [&](const owl::vec_t<float, Dim> &r_vec) -> owl::vec_t<float, Dim> {
        float divisor = Dim == 2 ? owl::length(r_vec) : owl::length2(r_vec);
        divisor = max(divisor, config.dGdx_regularization);
        owl::vec_t<float, Dim> result = owl::normalize(r_vec) / (divisor * (2 * (Dim - 1) * M_PIf32));
        return owl::isfinite(result) ? result : utils::zero<owl::vec_t<float, Dim>>();
    };

    const auto estimate_pseudo_boundary_term = [&](const owl::vec_t<float, Dim> &x,
                                                   const owl::vec_t<float, Dim> &x_vel) -> owl::vec_t<float, Dim> {
        if (config.domain_type == DomainType::BoundedDomain) return utils::zero<owl::vec_t<float, Dim>>();
        utils::KahanSum<owl::vec_t<float, Dim>> sum;
        for (int s = 0; s < config.num_pseudo_boundary_samples_direct; s++) {
            auto [point, inv_pdf] = cdf_boundary_sample<Dim>(pseudo_boundary_mesh, rand_state);
            inv_pdf *= is_point_outside_solid(point.position);
            sum += dGdx(point.position - x) *
                   (inv_pdf * dot(owl::cast_dim<Dim>(owl::xfmNormal(
                                      pseudo_boundary_mesh.transform, pseudo_boundary_mesh.normal_buf[point.prim_id]
                                  )),
                                  get_velocity(point.position) - x_vel));
        }
        return sum.sum / (float)config.num_pseudo_boundary_samples_direct;
    };

    const auto estimate_boundary_term = [&](const owl::vec_t<float, Dim> &x,
                                            const owl::vec_t<float, Dim> &x_vel) -> owl::vec_t<float, Dim> {
        if (mesh.num_primitives == 0) return utils::zero<owl::vec_t<float, Dim>>();
        auto [point, inv_pdf] = cdf_boundary_sample<Dim>(mesh, rand_state);
        inv_pdf *= is_point_inside_simulation_box(point.position);
        const owl::vec_t<float, Dim> normal =
            owl::cast_dim<Dim>(owl::xfmNormal(mesh.transform, mesh.normal_buf[point.prim_id]));
        return inv_pdf * dot(normal, x_vel) * dGdx(point.position - x);
    };

    const auto estimate_velocity_source_term = [&](const owl::vec_t<float, Dim> &x) -> owl::vec_t<float, Dim> {
        utils::KahanSum<owl::vec_t<float, Dim>> sum;
        for (int s = 0; s < velocity_source_count; s++) {
            const owl::vec_t<float, Dim> source_position = owl::cast_dim<Dim>(velocity_source_buf[s]);
            sum += velocity_source_buf[s][Dim] * dGdx(x - source_position);
        }
        return sum.sum;
    };

    owl::vec_t<float, Dim> original_vel = get_velocity(original_position);

    owl::vec_t<float, Dim> pressure_grad_total =
        -(estimate_volume_term(original_position, original_vel) +
          estimate_pseudo_boundary_term(original_position, original_vel) +
          estimate_velocity_source_term(original_position));

    utils::KahanSum<owl::vec_t<float, Dim>> pressure_grad_sum;

    if (mesh.num_primitives > 0) {
        for (int s = 0; s < config.num_path_samples; s++)
            pressure_grad_sum += estimate_boundary_term(original_position, original_vel);

        int data_count = config.num_path_samples * (config.path_length + 1);
        for (int s = 0; s < data_count; s++)
            pressure_grad_sum += vpl_data_buf[s].value * dGdx(vpl_data_buf[s].position - original_position);

        pressure_grad_total += pressure_grad_sum.sum / (float)config.num_path_samples;
    }

    return original_vel - pressure_grad_total;
}

OPTIX_RAYGEN_PROGRAM(velocityProject2DRayGen)() {
    const ProjectionRayGenData<2> &self = owl::getProgramData<ProjectionRayGenData<2>>();
    const owl::vec3i idx_3d = optixGetLaunchIndex();

    // round robin scheduling with 32 x indices per chunk.
    const int responsible_device = (idx_3d.x >> 5) % self.device_count;
    if (self.device_index != responsible_device) return;

    const owl::vec3i dim_3d = optixGetLaunchDimensions();
    const int idx = utils::flatten(idx_3d, dim_3d);
    if (idx >= self.num_evaluation_points) return;

    const owl::vec2f position = self.evaluation_point_buf[idx];
    utils::randState_t rand_state = self.random_state_buf[idx];

    owl::vec2f new_velocity = self.solid_velocity;
    float blend_weight = self.config.domain_type == DomainType::BoundedDomain ? self.winding_number_buf[idx]
                                                                              : 1.f + self.winding_number_buf[idx];
    if (blend_weight > 0.0f)
        new_velocity = (1.f - blend_weight) * new_velocity +
                       blend_weight * project(

                                          position, rand_state, self.grid_winding_number_buf, self.grid_velocity_buf,
                                          self.grid_advection_velocity_buf, self.solid_velocity,
                                          self.velocity_source_buf, self.velocity_source_count, self.dt,
                                          self.acc_structure, self.mesh, self.pseudo_boundary_mesh, self.config
                                      );

    self.new_velocity_buf[idx] = new_velocity;
    self.random_state_buf[idx] = rand_state;
}

OPTIX_RAYGEN_PROGRAM(velocityProject3DRayGen)() {
    const ProjectionRayGenData<3> &self = owl::getProgramData<ProjectionRayGenData<3>>();
    const owl::vec3i idx_3d = optixGetLaunchIndex();

    // round robin scheduling with 32 x indices per chunk.
    const int responsible_device = (idx_3d.x >> 5) % self.device_count;
    if (self.device_index != responsible_device) return;

    const owl::vec3i dim_3d = optixGetLaunchDimensions();
    const int idx = utils::flatten(idx_3d, dim_3d);
    if (idx >= self.num_evaluation_points) return;

    const owl::vec3f position = self.evaluation_point_buf[idx];
    utils::randState_t rand_state = self.random_state_buf[idx];

    owl::vec3f new_velocity = self.solid_velocity;
    float blend_weight = self.config.domain_type == DomainType::BoundedDomain ? self.winding_number_buf[idx]
                                                                              : 1.f + self.winding_number_buf[idx];
    if (blend_weight > 0.0f)
        new_velocity = (1.f - blend_weight) * new_velocity +
                       blend_weight * project(
                                          position, rand_state, self.grid_winding_number_buf, self.grid_velocity_buf,
                                          self.grid_advection_velocity_buf, self.solid_velocity,
                                          self.velocity_source_buf, self.velocity_source_count, self.dt,
                                          self.acc_structure, self.mesh, self.pseudo_boundary_mesh, self.config
                                      );

    self.new_velocity_buf[idx] = new_velocity;
    self.random_state_buf[idx] = rand_state;
}

OPTIX_RAYGEN_PROGRAM(velocityProjectVPLConstruct2DRayGen)() {
    const ProjectionRayGenData<2> &self = owl::getProgramData<ProjectionRayGenData<2>>();
    const owl::vec3i idx_3d = optixGetLaunchIndex();

    // round robin scheduling with 32 x indices per chunk.
    const int responsible_device = (idx_3d.x >> 5) % self.device_count;
    if (self.device_index != responsible_device) return;

    const owl::vec3i dim_3d = optixGetLaunchDimensions();
    const int idx = utils::flatten(idx_3d, dim_3d);
    if (idx >= self.config.num_path_samples) return;

    utils::randState_t rand_state = self.random_state_buf[idx];

    project_vpl_construct(
        self.vpl_data_buf + (self.config.path_length + 1) * idx, rand_state, self.grid_winding_number_buf,
        self.grid_velocity_buf, self.grid_advection_velocity_buf, self.solid_velocity, self.velocity_source_buf,
        self.velocity_source_count, self.dt, self.acc_structure, self.mesh, self.pseudo_boundary_mesh, self.config
    );

    self.random_state_buf[idx] = rand_state;
}

OPTIX_RAYGEN_PROGRAM(velocityProjectVPLGather2DRayGen)() {
    const ProjectionRayGenData<2> &self = owl::getProgramData<ProjectionRayGenData<2>>();
    const owl::vec3i idx_3d = optixGetLaunchIndex();

    // round robin scheduling with 32 x indices per chunk.
    const int responsible_device = (idx_3d.x >> 5) % self.device_count;
    if (self.device_index != responsible_device) return;

    const owl::vec3i dim_3d = optixGetLaunchDimensions();
    const int idx = utils::flatten(idx_3d, dim_3d);
    if (idx >= self.num_evaluation_points) return;

    const owl::vec2f position = self.evaluation_point_buf[idx];
    utils::randState_t rand_state = self.random_state_buf[idx];

    owl::vec2f new_velocity = self.solid_velocity;
    float blend_weight = self.config.domain_type == DomainType::BoundedDomain ? self.winding_number_buf[idx]
                                                                              : 1.f + self.winding_number_buf[idx];
    if (blend_weight > 0.0f)
        new_velocity = (1.f - blend_weight) * new_velocity +
                       blend_weight * project_vpl_gather(
                                          position, rand_state, self.grid_winding_number_buf, self.grid_velocity_buf,
                                          self.grid_advection_velocity_buf, self.solid_velocity,
                                          self.velocity_source_buf, self.velocity_source_count, self.vpl_data_buf,
                                          self.dt, self.acc_structure, self.mesh, self.pseudo_boundary_mesh, self.config
                                      );

    self.new_velocity_buf[idx] = new_velocity;
    self.random_state_buf[idx] = rand_state;
}

OPTIX_RAYGEN_PROGRAM(velocityProjectVPLConstruct3DRayGen)() {
    const ProjectionRayGenData<3> &self = owl::getProgramData<ProjectionRayGenData<3>>();
    const owl::vec3i idx_3d = optixGetLaunchIndex();

    // round robin scheduling with 32 x indices per chunk.
    const int responsible_device = (idx_3d.x >> 5) % self.device_count;
    if (self.device_index != responsible_device) return;

    const owl::vec3i dim_3d = optixGetLaunchDimensions();
    const int idx = utils::flatten(idx_3d, dim_3d);
    if (idx >= self.config.num_path_samples) return;

    utils::randState_t rand_state = self.random_state_buf[idx];

    project_vpl_construct(
        self.vpl_data_buf + (self.config.path_length + 1) * idx, rand_state, self.grid_winding_number_buf,
        self.grid_velocity_buf, self.grid_advection_velocity_buf, self.solid_velocity, self.velocity_source_buf,
        self.velocity_source_count, self.dt, self.acc_structure, self.mesh, self.pseudo_boundary_mesh, self.config
    );

    self.random_state_buf[idx] = rand_state;
}

OPTIX_RAYGEN_PROGRAM(velocityProjectVPLGather3DRayGen)() {
    const ProjectionRayGenData<3> &self = owl::getProgramData<ProjectionRayGenData<3>>();
    const owl::vec3i idx_3d = optixGetLaunchIndex();

    // round robin scheduling with 32 x indices per chunk.
    const int responsible_device = (idx_3d.x >> 5) % self.device_count;
    if (self.device_index != responsible_device) return;

    const owl::vec3i dim_3d = optixGetLaunchDimensions();
    const int idx = utils::flatten(idx_3d, dim_3d);
    if (idx >= self.num_evaluation_points) return;

    const owl::vec3f position = self.evaluation_point_buf[idx];
    utils::randState_t rand_state = self.random_state_buf[idx];

    owl::vec3f new_velocity = self.solid_velocity;
    float blend_weight = self.config.domain_type == DomainType::BoundedDomain ? self.winding_number_buf[idx]
                                                                              : 1.f + self.winding_number_buf[idx];
    if (blend_weight > 0.0f)
        new_velocity = (1.f - blend_weight) * new_velocity +
                       blend_weight * project_vpl_gather(
                                          position, rand_state, self.grid_winding_number_buf, self.grid_velocity_buf,
                                          self.grid_advection_velocity_buf, self.solid_velocity,
                                          self.velocity_source_buf, self.velocity_source_count, self.vpl_data_buf,
                                          self.dt, self.acc_structure, self.mesh, self.pseudo_boundary_mesh, self.config
                                      );

    self.new_velocity_buf[idx] = new_velocity;
    self.random_state_buf[idx] = rand_state;
}