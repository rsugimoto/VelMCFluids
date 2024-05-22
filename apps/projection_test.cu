// Projection error test.

#include "common.cuh"
#include "geometry.cuh"
#include "utils.hpp"
#include <cmath>
#include <filesystem>
#include <thrust/universal_vector.h>

__host__ __device__ owl::vec2f preprojection_velocity(const owl::vec2f &pos) {
    const owl::vec2f c1(-0.15f, 0.0f);
    const owl::vec2f c2(0.15f, 0.0f);
    const float R = 0.3f;

    // owl::vec2f vel(0.1f, 0.0f);
    // for (const owl::vec2f &c : {c1, c2}) {
    //     owl::vec2f r_vec = pos - c;
    //     float r = owl::length(r_vec);
    //     if (r <= R / 2.f) {
    //         vel += r * owl::normalize(owl::vec2f(-r_vec.y, r_vec.x));
    //         vel += r * owl::normalize(r_vec);
    //     } else if (r <= R) {
    //         vel += (R - r) * owl::normalize(owl::vec2f(-r_vec.y, r_vec.x));
    //         vel += (R - r) * owl::normalize(r_vec);
    //     }
    // }

    owl::vec2f vel = min(owl::length(pos), R) * owl::normalize(pos);
    return vel;
}

// Ground truth
__host__ __device__ owl::vec2f postprojection_velocity(const owl::vec2f &pos) {
    const owl::vec2f c1(-0.15f, 0.0f);
    const owl::vec2f c2(0.15f, 0.0f);
    const float R = 0.3f;

    // owl::vec2f vel(0.1f, 0.0f);
    // for (const owl::vec2f &c : {c1, c2}) {
    //     owl::vec2f r_vec = pos - c;
    //     float r = owl::length(r_vec);
    //     // divergence free component
    //     if (r <= R / 2.f) {
    //         vel += r * owl::normalize(owl::vec2f(-r_vec.y, r_vec.x));
    //     } else if (r <= R) {
    //         vel += (R - r) * owl::normalize(owl::vec2f(-r_vec.y, r_vec.x));
    //     }
    // }

    owl::vec2f vel = min(owl::length(pos), R) * owl::normalize(pos);
    return vel;
}

inline __device__ float max_domain_corner_distance(const owl::vec2f &position, const owl::vec2f &domain_size) {
    float sampling_radius2 =
        max(max(owl::length2(position - owl::vec2f(-domain_size.x / 2, -domain_size.y / 2)),
                owl::length2(position - owl::vec2f(-domain_size.x / 2, +domain_size.y / 2))),
            max(owl::length2(position - owl::vec2f(+domain_size.x / 2, -domain_size.y / 2)),
                owl::length2(position - owl::vec2f(+domain_size.x / 2, +domain_size.y / 2))));
    return sqrtf(sampling_radius2);
}

inline __device__ VolumeSample<2> uniform_domain_sample(const owl::vec2f &domain_size, utils::randState_t &rand_state) {
    owl::vec2f pos = owl::vec2f(
        utils::rand_uniform(rand_state) * domain_size.x - domain_size.x / 2,
        utils::rand_uniform(rand_state) * domain_size.y - domain_size.y / 2
    );
    return {pos, domain_size.x * domain_size.y};
}

// The first two elements are velocity components, the third element is the pressure.
__device__ owl::vec3f project(
    const owl::vec2f &pos, utils::randState_t &rand_state, const Mesh &pseudo_boundary_mesh,
    const float num_volume_samples, const float num_pseudo_boundary_samples, const bool enable_antithetic_sampling,
    const bool enable_vel_shift, const float dGdx_regularization, const owl::vec2f &domain_size,
    const bool uniform_sampling = false
) {
    const auto estimate_volume_term = [&](const owl::vec2f &x, const owl::vec2f &x_vel,
                                          const int num_samples) -> owl::vec3f {
        const owl::vec2f half_domain_size = domain_size / 2.f;
        float sampling_radius = max_domain_corner_distance(x, domain_size);
        utils::KahanSum<owl::vec2f> S_sum;
        utils::KahanSum<float> dGdy_sum;
        for (int v = 0; v < num_samples; v++) {
            auto [r_vec, inv_pdf] = uniform_sampling ? uniform_domain_sample(domain_size, rand_state)
                                                     : strongly_singular_ball_sample<2>(sampling_radius, rand_state);
            const float r2 = owl::length2(r_vec);
            const float r = owl::length(r_vec);
            owl::vec2f r_hat = owl::normalize(r_vec);

            if (enable_antithetic_sampling) v += 1; // count the second sample.
            for (int i = 0; i < (int)enable_antithetic_sampling + 1; i++) {
                if (i == 1) {
                    r_vec = -r_vec;
                    r_hat = -r_hat;
                }
                const owl::vec2f y = x + r_vec;
                const owl::vec2f vel_diff =
                    enable_vel_shift ? preprojection_velocity(y) - x_vel : preprojection_velocity(y);
                if ((abs(y.x) <= half_domain_size.x && abs(y.y) <= half_domain_size.y)) {
                    S_sum += inv_pdf / r2 * (2 * dot(r_hat, vel_diff) * r_hat - vel_diff);
                    dGdy_sum += inv_pdf / r * dot(r_hat, vel_diff); // use plus here because it's gonna be flipped
                                                                    // later. So here, it is actually computing -dGdy.
                }
            }
        }
        return owl::vec3f(S_sum.sum.x, S_sum.sum.y, dGdy_sum.sum) / (num_samples * (2 * (2 - 1) * M_PIf32));
    };

    const auto dGdx = [&](const owl::vec2f &r_vec) -> owl::vec3f {
        float r = owl::length(r_vec);
        r = max(r, dGdx_regularization);
        owl::vec2f _dGdx = owl::normalize(r_vec) / (r * (2 * (2 - 1) * M_PIf32));
        if (!owl::isfinite(_dGdx)) _dGdx = utils::zero<owl::vec2f>();
        float _G = -1. / (2 * M_PIf32) * log(r);
        if (!owl::isfinite(_G)) _G = 0.f;

        return owl::vec3f(_dGdx.x, _dGdx.y, _G);
    };

    const auto estimate_pseudo_boundary_term = [&](const owl::vec2f &x, const owl::vec2f &x_vel,
                                                   const int num_samples) -> owl::vec3f {
        utils::KahanSum<owl::vec3f> sum;
        for (int s = 0; s < num_samples; s++) {
            const auto [point, inv_pdf] = cdf_boundary_sample<2>(pseudo_boundary_mesh, rand_state);
            sum += dGdx(point.position - x) *
                   (inv_pdf * dot(owl::cast_dim<2>(owl::xfmNormal(
                                      pseudo_boundary_mesh.transform, pseudo_boundary_mesh.normal_buf[point.prim_id]
                                  )),
                                  enable_vel_shift ? preprojection_velocity(point.position) - x_vel
                                                   : preprojection_velocity(point.position)));
        }
        return sum.sum / (float)num_samples;
    };

    owl::vec2f original_vel = preprojection_velocity(pos);

    owl::vec3f pressure_grad_total =
        -(estimate_volume_term(pos, original_vel, num_volume_samples) +
          estimate_pseudo_boundary_term(pos, original_vel, num_pseudo_boundary_samples));
    if (!enable_vel_shift) {
        pressure_grad_total.x += 0.5f * original_vel.x;
        pressure_grad_total.y += 0.5f * original_vel.y;
    }

    return owl::vec3f(
        original_vel.x - pressure_grad_total.x, original_vel.y - pressure_grad_total.y, pressure_grad_total.z
    );
}

int main(int argc, char *argv[]) {
    const owl::vec2f domain_size(1.0f, 1.0f);
    const owl::vec2i grid_res(256, 256);
    const float dGdx_regularization = 1e-5f;
    const std::string output_dir = "../results/projection_test/raw";

    std::filesystem::create_directories(output_dir);

    int num_evaluation_points = grid_res.x * grid_res.y;
    thrust::universal_vector<owl::vec2f> velocity(num_evaluation_points);
    thrust::universal_vector<float> pressure(num_evaluation_points);
    thrust::universal_vector<float> squared_error(num_evaluation_points);
    thrust::universal_vector<utils::randState_t> random_state(num_evaluation_points);
    owl::vec2f *velocity_ptr = velocity.data().get();
    float *pressure_ptr = pressure.data().get();
    float *squared_error_ptr = squared_error.data().get();
    utils::randState_t *random_state_ptr = random_state.data().get();

    thrust::universal_vector<owl::vec3f> pseudo_boundary_vertex_buf;
    thrust::universal_vector<owl::vec3i> pseudo_boundary_index_buf;
    thrust::universal_vector<owl::vec3f> pseudo_boundary_normal_buf;
    thrust::universal_vector<float> pseudo_boundary_area_buf;
    thrust::universal_vector<float> pseudo_boundary_area_cdf_buf;
    int pseudo_boundary_num_primitives = 0;
    {
        std::vector<owl::vec3f> pseudo_boundary_vertices;
        std::vector<owl::vec3i> pseudo_boundary_indices;
        std::vector<owl::vec3f> pseudo_boundary_normals;
        std::vector<float> pseudo_boundary_areas;
        std::vector<float> pseudo_boundary_area_cdf;

        owl::vec2f vertices[4] = {
            {domain_size[0] / 2, domain_size[1] / 2},
            {domain_size[0] / 2, -domain_size[1] / 2},
            {-domain_size[0] / 2, -domain_size[1] / 2},
            {-domain_size[0] / 2, domain_size[1] / 2}};
        for (int l = 0; l < 4; l++) {
            const owl::vec2f &v0 = vertices[l];
            const owl::vec2f &v1 = vertices[(l + 1) % 4];
            pseudo_boundary_vertices.emplace_back(v0.x, v0.y, 0.f);
            pseudo_boundary_vertices.emplace_back(v1.x, v1.y, -1.f);
            pseudo_boundary_vertices.emplace_back(v1.x, v1.y, 1.f);
            pseudo_boundary_indices.emplace_back(
                pseudo_boundary_vertices.size() - 3, pseudo_boundary_vertices.size() - 2,
                pseudo_boundary_vertices.size() - 1
            );
            const owl::vec2f t = v1 - v0;
            const owl::vec2f n = owl::normalize(owl::vec2f(-t.y, t.x));
            pseudo_boundary_normals.emplace_back(owl::cast_dim<3>(n));
            pseudo_boundary_areas.emplace_back(owl::length(t));
        }

        pseudo_boundary_num_primitives = pseudo_boundary_indices.size();
        pseudo_boundary_area_cdf.resize(pseudo_boundary_num_primitives);
        double area_sum = 0.f;
        for (int f = 0; f < pseudo_boundary_num_primitives; f++) {
            area_sum += pseudo_boundary_areas[f];
            pseudo_boundary_area_cdf[f] = area_sum;
        }
        for (int f = 0; f < pseudo_boundary_num_primitives; f++) pseudo_boundary_area_cdf[f] /= area_sum;

        pseudo_boundary_vertex_buf = pseudo_boundary_vertices;
        pseudo_boundary_index_buf = pseudo_boundary_indices;
        pseudo_boundary_normal_buf = pseudo_boundary_normals;
        pseudo_boundary_area_buf = pseudo_boundary_areas;
        pseudo_boundary_area_cdf_buf = pseudo_boundary_area_cdf;
    }
    Mesh pseudo_boundary_mesh{pseudo_boundary_vertex_buf.data().get(), pseudo_boundary_index_buf.data().get(),
                              pseudo_boundary_normal_buf.data().get(), owl::affine3f(),
                              pseudo_boundary_area_buf.data().get(),   pseudo_boundary_area_cdf_buf.data().get(),
                              pseudo_boundary_num_primitives};

    {
        thrust::for_each(
            thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(num_evaluation_points),
            [=] __device__(int idx) {
                owl::vec2f pos = utils::idx_to_domain_point<2>(idx, grid_res, domain_size);
                velocity_ptr[idx] = preprojection_velocity(pos);
            }
        );
        utils::save_field(velocity, output_dir + "/velocity_preproject.vector", grid_res);
        thrust::for_each(
            thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(num_evaluation_points),
            [=] __device__(int idx) {
                owl::vec2f pos = utils::idx_to_domain_point<2>(idx, grid_res, domain_size);
                velocity_ptr[idx] = postprojection_velocity(pos);
            }
        );
        utils::save_field(velocity, output_dir + "/velocity_postproject.vector", grid_res);
    }

    bool configs[4][3] = {{true, true, false}, {false, true, false}, {true, false, false}, {true, true, true}};

    for (bool *config : configs) {
        const bool enable_antithetic_sampling = config[0];
        const bool enable_vel_shift = config[1];
        const bool use_uniform_sampling = config[2];
        std::cout << "enable_antithetic_sampling: " << enable_antithetic_sampling << std::endl;
        std::cout << "enable_vel_shift: " << enable_vel_shift << std::endl;
        std::cout << "use_uniform_sampling: " << use_uniform_sampling << std::endl;

        int num_volume_samples = 1e7;
        int num_pseudo_boundary_samples = 1e7;
        for (int num_volume_samples : {1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7}) {
            utils::random_states_init(random_state);
            thrust::for_each(
                thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(num_evaluation_points),
                [=] __device__(int idx) {
                    owl::vec2f pos = utils::idx_to_domain_point<2>(idx, grid_res, domain_size);
                    owl::vec3f result = project(
                        pos, random_state_ptr[idx], pseudo_boundary_mesh, num_volume_samples,
                        num_pseudo_boundary_samples, enable_antithetic_sampling, enable_vel_shift, dGdx_regularization,
                        domain_size, use_uniform_sampling
                    );
                    velocity_ptr[idx] = owl::vec2f(result.x, result.y);
                    pressure_ptr[idx] = result.z;
                    squared_error_ptr[idx] = owl::length2(velocity_ptr[idx] - postprojection_velocity(pos));
                }
            );
            float RMSE = sqrtf(thrust::reduce(squared_error.begin(), squared_error.end(), 0.f, thrust::plus<float>()));
            std::cout << "num_volume_samples: " << num_volume_samples << " " << RMSE << std::endl;
            utils::save_field(
                velocity, output_dir + "/velocity_projected_vol" + std::to_string(num_volume_samples) + ".vector",
                grid_res
            );
            utils::save_field(
                pressure, output_dir + "/pressure_projected_vol" + std::to_string(num_volume_samples) + ".scalar",
                grid_res
            );
        }

        for (int num_pseudo_boundary_samples : {1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7}) {
            utils::random_states_init(random_state);
            thrust::for_each(
                thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(num_evaluation_points),
                [=] __device__(int idx) {
                    owl::vec2f pos = utils::idx_to_domain_point<2>(idx, grid_res, domain_size);
                    owl::vec3f result = project(
                        pos, random_state_ptr[idx], pseudo_boundary_mesh, num_volume_samples,
                        num_pseudo_boundary_samples, enable_antithetic_sampling, enable_vel_shift, dGdx_regularization,
                        domain_size, use_uniform_sampling
                    );
                    velocity_ptr[idx] = owl::vec2f(result.x, result.y);
                    pressure_ptr[idx] = result.z;
                    squared_error_ptr[idx] = owl::length2(velocity_ptr[idx] - postprojection_velocity(pos));
                }
            );
            float RMSE = sqrtf(thrust::reduce(squared_error.begin(), squared_error.end(), 0.f, thrust::plus<float>()));
            std::cout << "num_pseudo_boundary_samples: " << num_pseudo_boundary_samples << " " << RMSE << std::endl;
            utils::save_field(
                velocity,
                output_dir + "/velocity_projected_boundary" + std::to_string(num_pseudo_boundary_samples) + ".vector",
                grid_res
            );
            utils::save_field(
                pressure, output_dir + "/pressure_projected_vol" + std::to_string(num_volume_samples) + ".scalar",
                grid_res
            );
        }
    }

    return 0;
}