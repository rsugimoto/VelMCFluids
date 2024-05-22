// Standard Advection-Projection method with optional diffusion.
// This version uses a PIC/FLIP style particle grid hybrid to advect the velocity field.
// The projection and adveciton are comined into one step in this version.

#include "utils.hpp"
#include "velocity_fluids_host.cuh"
#include <filesystem>
#include <iostream>

constexpr int Dim = 2;

int main(int argc, char *argv[]) {
    const char *input_json_file = argv[1];
    Config<Dim> config = load_config<Dim>(input_json_file);

    float flip_blend_ratio = config.json_config["flip_blend_ratio"].get<float>();
    int num_particles = config.num_evaluation_points * 4 * (Dim - 1);

    const float scene_two_domain_extend = 1.2f;
    if (config.scene_number == 2)
        num_particles *= 1.2; // For Karman vortex street scene, we use particles slightly outside the domain,
                              // too, to avoid having no particles in some cells.

    std::cout << "flip blend ratio\t:" << flip_blend_ratio << std::endl;
    std::cout << "num particles\t:" << num_particles << std::endl;

    std::filesystem::create_directories(config.output_dir);
    std::filesystem::copy(
        input_json_file, config.output_dir + "/config.json", std::filesystem::copy_options::overwrite_existing
    );

    Scene<Dim> scene(config);

    std::cout << "Initializing buffers..." << std::endl;
    thrust::universal_vector<owl::vec_t<float, Dim>> grid_evaluation_point(config.num_evaluation_points);
    scene.initialize_evaluation_points(grid_evaluation_point);

    thrust::universal_vector<float> grid_winding_number(config.num_evaluation_points);
    thrust::universal_vector<owl::vec_t<float, Dim>> grid_velocity(config.num_evaluation_points);
    thrust::universal_vector<float> grid_weight_sum(config.num_evaluation_points);
    thrust::universal_vector<utils::randState_t> grid_random_state(config.num_evaluation_points);
    utils::random_states_init(grid_random_state);

    thrust::universal_vector<float> grid_concentration(config.num_evaluation_points);
    thrust::universal_vector<float> grid_temperature(config.num_evaluation_points);

    std::mt19937_64 cpu_random;
    std::uniform_real_distribution<float> cpu_dist(-0.5f, 0.5f);

    thrust::universal_vector<owl::vec_t<float, Dim>> particle_positions(num_particles);
    {
        for (auto &particle_position : particle_positions) {
            if (Dim == 2)
                particle_position = owl::cast_dim<Dim>(owl::vec2f(
                    cpu_dist(cpu_random) * config.domain_size[0], cpu_dist(cpu_random) * config.domain_size[1]
                ));
            else if (Dim == 3)
                particle_position = owl::cast_dim<Dim>(owl::vec3f(
                    cpu_dist(cpu_random) * config.domain_size[0], cpu_dist(cpu_random) * config.domain_size[1],
                    cpu_dist(cpu_random) * config.domain_size[2]
                ));

            if (config.scene_number == 2) particle_position.y *= scene_two_domain_extend;
        }
    }

    thrust::universal_vector<utils::randState_t> particle_random_states(num_particles);
    utils::random_states_init(particle_random_states);

    thrust::universal_vector<owl::vec_t<float, Dim>> particle_velocities(num_particles);
    thrust::universal_vector<float> particle_concentrations(num_particles);
    thrust::universal_vector<float> particle_temperatures(num_particles);

    thrust::universal_vector<owl::vec_t<float, Dim>> particle_temp_positions(num_particles);
    thrust::universal_vector<float> particle_winding_numbers(num_particles);
    thrust::universal_vector<owl::vec_t<float, Dim>> particle_velocities_k1(num_particles);
    thrust::universal_vector<owl::vec_t<float, Dim>> particle_velocities_k2(num_particles);
    thrust::universal_vector<owl::vec_t<float, Dim>> particle_velocities_k3(num_particles);
    thrust::universal_vector<owl::vec_t<float, Dim>> particle_velocities_k4(num_particles);

    auto &particle_temp_velocities = particle_velocities_k1;
    thrust::universal_vector<float> particle_temp_concentrations(num_particles);
    thrust::universal_vector<float> particle_temp_temperatures(num_particles);

    thrust::universal_vector<ProjectionVPLRecord<Dim>> vpl_data;
    thrust::universal_vector<utils::randState_t> vpl_random_state;

    if (config.enable_vpl) {
        vpl_data.resize(config.num_path_samples * (config.path_length + 1));
        vpl_random_state.resize(config.num_path_samples);
        utils::random_states_init(vpl_random_state);
    }

    // Set initial conditions
    {
        scene.winding_number(grid_evaluation_point, grid_winding_number, grid_random_state);
        owl::vec_t<float, Dim> *grid_velocity_ptr = grid_velocity.data().get();
        float *grid_concentration_ptr = grid_concentration.data().get();
        float *grid_temperature_ptr = grid_temperature.data().get();
        const float *winding_number_ptr = grid_winding_number.data().get();
        owl::vec_t<int, Dim> grid_res = config.grid_res;
        owl::vec_t<float, Dim> domain_size = config.domain_size;
        DomainType domain_type = config.domain_type;
        owl::vec_t<float, Dim> solid_velocity = config.solid_velocity;
        int scene_number = config.scene_number;
        thrust::for_each(
            thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(config.num_evaluation_points),
            [=] __device__(int idx) {
                owl::vec_t<float, Dim> pos = utils::idx_to_domain_point<Dim>(idx, grid_res, domain_size);
                grid_velocity_ptr[idx] = owl::vec_t<float, Dim>(0.0, 0.0);
                grid_concentration_ptr[idx] = 0.0;
                grid_temperature_ptr[idx] = 0.0;

                switch (scene_number) {
                case 0: // zero initial conditions
                    break;
                case 1: // circulating velocity
                {
                    float r = owl::length(pos);
                    float magnitude = max(min(r, 0.95f - r), 0.f); // r > 0.95f ? 0.0f : r;
                    owl::vec_t<float, Dim> direction = owl::normalize(owl::vec_t<float, Dim>(pos.y, -pos.x));
                    grid_velocity_ptr[idx] = magnitude * direction;
                    break;
                }
                case 2: // karman vortex street
                {
                    owl::vec_t<int, Dim> idx_md = utils::unflatten(idx, grid_res);
                    grid_velocity_ptr[idx] = owl::vec_t<float, Dim>(1.f, 0.f);
                    break;
                }
                default: break;
                }

                // Point inside the solid obstacles should have the solid velocity.
                float blend_weight =
                    domain_type == DomainType::BoundedDomain ? winding_number_ptr[idx] : 1.0f + winding_number_ptr[idx];
                grid_velocity_ptr[idx] = blend_weight * grid_velocity_ptr[idx] + (1.0f - blend_weight) * solid_velocity;
            }
        );
    }

    utils::save_field(grid_winding_number, config.output_dir + "/winding_number_0.scalar", config.grid_res);
    utils::save_field(grid_velocity, config.output_dir + "/velocity_preproject_0.vector", config.grid_res);
    utils::save_field(grid_concentration, config.output_dir + "/concentration_0.scalar", config.grid_res);
    utils::save_field(grid_temperature, config.output_dir + "/temperature_0.scalar", config.grid_res);

    std::cout << "Starting time stepping... " << std::endl;

    // Step 0 is without advection and diffusion. We do this so the first diffusion step will still use an
    // incompressible velocity field.
    {
        auto start_time = std::chrono::high_resolution_clock::now();
        thrust::swap(grid_velocity, grid_velocity);
        scene.winding_number(particle_positions, particle_winding_numbers, particle_random_states);
        scene.project(
            particle_positions, particle_winding_numbers, particle_velocities, particle_random_states,
            grid_winding_number, grid_velocity, grid_velocity, vpl_data, vpl_random_state, config.dt, false,
            config.velocity_sources
        );

        thrust::fill(particle_concentrations.begin(), particle_concentrations.end(), 0.0f);
        thrust::fill(particle_temperatures.begin(), particle_temperatures.end(), 0.0f);

        scene.particle_to_grid(particle_positions, particle_velocities, grid_velocity, grid_weight_sum);
        utils::save_field(grid_velocity, config.output_dir + "/velocity_0.vector", config.grid_res);
        auto stop_time = std::chrono::high_resolution_clock::now();

        std::cout << std::chrono::high_resolution_clock::now() << ": step 0, t=0 ("
                  << (double)std::chrono::duration_cast<std::chrono::milliseconds>(stop_time - start_time).count() /
                         1000.0
                  << "s)" << std::endl;
    }

    bool enable_concentration_diffusion = config.concentration_diffusion_coefficient > 0;
    bool enable_temperature_diffusion = config.temperature_diffusion_coefficient > 0;
    bool enable_velocity_diffusion = config.velocity_diffusion_coefficient > 0;
    for (int step = 1; step <= config.time_steps; step++) {
        auto start_time = std::chrono::high_resolution_clock::now();

        scene.winding_number(particle_positions, particle_winding_numbers, particle_random_states);

        // diffuse
        if (enable_concentration_diffusion) {
            scene.diffuse_scalar(
                particle_positions, particle_winding_numbers, particle_temp_concentrations, particle_random_states,
                grid_winding_number, grid_concentration, grid_velocity, config.concentration_diffusion_coefficient,
                config.dt, false
            );
            for (int idx = 0; idx < num_particles; idx++) {
                const float old_concentration = utils::sample_buffer_linear(
                    utils::domain_point_to_idx(particle_positions[idx], config.grid_res, config.domain_size),
                    grid_concentration.data().get(), config.grid_res, utils::WrapMode::ClampToEdge
                );
                particle_concentrations[idx] += particle_temp_concentrations[idx] - old_concentration;
            }
            scene.particle_to_grid(particle_positions, particle_concentrations, grid_concentration, grid_weight_sum);
        }

        if (enable_temperature_diffusion) {
            scene.diffuse_scalar(
                particle_positions, particle_winding_numbers, particle_temp_temperatures, particle_random_states,
                grid_winding_number, grid_temperature, grid_velocity, config.temperature_diffusion_coefficient,
                config.dt, false
            );
            for (int idx = 0; idx < num_particles; idx++) {
                const float old_temperature = utils::sample_buffer_linear(
                    utils::domain_point_to_idx(particle_positions[idx], config.grid_res, config.domain_size),
                    grid_temperature.data().get(), config.grid_res, utils::WrapMode::ClampToEdge
                );
                particle_temperatures[idx] += particle_temp_temperatures[idx] - old_temperature;
            }
            scene.particle_to_grid(particle_positions, particle_temperatures, grid_temperature, grid_weight_sum);
        }

        if (enable_velocity_diffusion) {
            scene.diffuse_velocity(
                particle_positions, particle_winding_numbers, particle_temp_velocities, particle_random_states,
                grid_winding_number, grid_velocity, grid_velocity, config.velocity_diffusion_coefficient, config.dt,
                false
            );
            for (int idx = 0; idx < num_particles; idx++) {
                const owl::vec_t<float, Dim> old_velocity = utils::sample_buffer_linear(
                    utils::domain_point_to_idx(particle_positions[idx], config.grid_res, config.domain_size),
                    grid_velocity.data().get(), config.grid_res, utils::WrapMode::ClampToEdge
                );
                particle_velocities[idx] += particle_temp_velocities[idx] - old_velocity;
            }
            scene.particle_to_grid(particle_positions, particle_velocities, grid_velocity, grid_weight_sum);
        }

        // for buoyancy simulation, add concentration and temperature source and apply the acceleration
        if (config.buoyancy_alpha != 0.0 || config.buoyancy_beta != 0.0) {
            for (int idx = 0; idx < num_particles; idx++) {
                // These are hardcoded.
                const owl::vec_t<float, Dim> pos = particle_positions[idx];
                // add concentration and temperature box source
                if (abs(pos.x - 0.0625f) < 0.125f && abs(pos.y - -0.75f) < 0.125f) {
                    particle_concentrations[idx] =
                        min(particle_concentrations[idx] + config.dt * config.concentration_rate, 1.0f);
                    particle_temperatures[idx] +=
                        (1.0 - std::exp(-config.temperature_rate * config.dt)) * (1.0 - particle_temperatures[idx]);
                }

                const owl::vec_t<float, Dim> acceleration = (config.buoyancy_alpha * particle_concentrations[idx] -
                                                             config.buoyancy_beta * particle_temperatures[idx]) *
                                                            config.buoyancy_gravity;
                particle_velocities[idx] += config.dt * acceleration;
            }
            scene.particle_to_grid(particle_positions, particle_velocities, grid_velocity, grid_weight_sum);
            scene.particle_to_grid(particle_positions, particle_concentrations, grid_concentration, grid_weight_sum);
            scene.particle_to_grid(particle_positions, particle_temperatures, grid_temperature, grid_weight_sum);
        }

        // move the solid obstacles
        if (config.solid_velocity != utils::zero<owl::vec_t<float, Dim>>()) {
            scene.setTranslation(config.obstacle_shift + step * config.dt * config.solid_velocity);
            scene.winding_number(grid_evaluation_point, grid_winding_number, grid_random_state);
            utils::save_field(
                grid_winding_number, config.output_dir + "/winding_number_" + std::to_string(step) + ".scalar",
                config.grid_res
            );
        }

        // project and advect all three fields
        {
            // If we are using VPL, use the same cache data for all substeps in advection.
            if (config.enable_vpl)
                scene.project_vpl_construct(
                    vpl_data, vpl_random_state, grid_winding_number, grid_velocity, grid_velocity, 0.0f, false,
                    config.velocity_sources
                );

            // Define this for convenience.
            auto project_temp_positions = [&](thrust::universal_vector<owl::vec_t<float, Dim>> &particle_new_velocities
                                          ) {
                scene.winding_number(particle_temp_positions, particle_winding_numbers, particle_random_states);
                if (config.enable_vpl)
                    scene.project_vpl_gather(
                        particle_temp_positions, particle_winding_numbers, particle_new_velocities,
                        particle_random_states, grid_winding_number, grid_velocity, grid_velocity, vpl_data, 0.0f,
                        false, config.velocity_sources
                    );
                else
                    scene.project_pointwise(
                        particle_temp_positions, particle_winding_numbers, particle_new_velocities,
                        particle_random_states, grid_winding_number, grid_velocity, grid_velocity, 0.0f, false,
                        config.velocity_sources
                    );
            };

            // We evaluate the velocity field at the original point first.
            particle_temp_positions = particle_positions;
            project_temp_positions(particle_velocities_k1);

            for (int idx = 0; idx < num_particles; idx++) {
                const owl::vec_t<float, Dim> particle_pos_old_velocity = utils::sample_buffer_linear(
                    utils::domain_point_to_idx(particle_positions[idx], config.grid_res, config.domain_size),
                    grid_velocity.data().get(), config.grid_res, utils::WrapMode::ClampToEdge
                );

                // The projected velocity for the particle location is computed using linear interpolation using
                // the formula above. Subtracting this velocity gives the FLIP velocity update, which is negative
                // prressure gradient.
                const owl::vec_t<float, Dim> flip_velocity =
                    particle_velocities[idx] + (particle_velocities_k1[idx] - particle_pos_old_velocity);

                particle_velocities[idx] =
                    flip_blend_ratio * flip_velocity + (1.0f - flip_blend_ratio) * particle_velocities_k1[idx];
            }

            switch (config.advection_mode) {
            case AdvectionMode::Euler: {
                for (int idx = 0; idx < num_particles; idx++)
                    particle_positions[idx] = particle_positions[idx] + config.dt * particle_velocities_k1[idx];
                break;
            }
            case AdvectionMode::MacCormack: {
                for (int idx = 0; idx < num_particles; idx++)
                    particle_temp_positions[idx] = particle_positions[idx] + config.dt * particle_velocities_k1[idx];
                project_temp_positions(particle_velocities_k2);

                for (int idx = 0; idx < num_particles; idx++)
                    particle_positions[idx] =
                        particle_positions[idx] +
                        0.5f * config.dt * (particle_velocities_k1[idx] + particle_velocities_k2[idx]);
                break;
            }
            case AdvectionMode::RK3: {
                for (int idx = 0; idx < num_particles; idx++)
                    particle_temp_positions[idx] =
                        particle_positions[idx] + 0.5f * config.dt * particle_velocities_k1[idx];
                project_temp_positions(particle_velocities_k2);

                for (int idx = 0; idx < num_particles; idx++)
                    particle_temp_positions[idx] =
                        particle_positions[idx] + 0.75f * config.dt * particle_velocities_k2[idx];
                project_temp_positions(particle_velocities_k3);

                for (int idx = 0; idx < num_particles; idx++)
                    particle_positions[idx] = particle_positions[idx] + config.dt *
                                                                            (2.f * particle_velocities_k1[idx] +
                                                                             3.f * particle_velocities_k2[idx] +
                                                                             4.f * particle_velocities_k3[idx]) /
                                                                            9.f;
                break;
            }
            case AdvectionMode::RK4: {
                for (int idx = 0; idx < num_particles; idx++)
                    particle_temp_positions[idx] =
                        particle_positions[idx] + 0.5f * config.dt * particle_velocities_k1[idx];
                project_temp_positions(particle_velocities_k2);

                for (int idx = 0; idx < num_particles; idx++)
                    particle_temp_positions[idx] =
                        particle_positions[idx] + 0.5f * config.dt * particle_velocities_k2[idx];
                project_temp_positions(particle_velocities_k3);

                for (int idx = 0; idx < num_particles; idx++)
                    particle_temp_positions[idx] = particle_positions[idx] + config.dt * particle_velocities_k3[idx];
                project_temp_positions(particle_velocities_k4);

                for (int idx = 0; idx < num_particles; idx++)
                    particle_positions[idx] = particle_positions[idx] +
                                              config.dt *
                                                  (particle_velocities_k1[idx] + 2.f * particle_velocities_k2[idx] +
                                                   2.f * particle_velocities_k3[idx] + particle_velocities_k4[idx]) /
                                                  6.f;

                break;
            }
            }

            // reseeding for the Karman vortex street scene
            if (config.scene_number == 2) {
                const float max_travel_dist = 1.f * config.dt;
                for (int idx = 0; idx < num_particles; idx++) {
                    owl::vec_t<float, Dim> &pos = particle_positions[idx];
                    if (pos[0] < config.domain_size[0] * 0.5f) continue;

                    pos = owl::cast_dim<Dim>(owl::vec2f(
                        (cpu_dist(cpu_random) + 0.5f) * max_travel_dist - 0.5f * config.domain_size[0],
                        cpu_dist(cpu_random) * config.domain_size[1] * scene_two_domain_extend
                    ));

                    particle_velocities[idx] = owl::vec_t<float, Dim>(1.0f, 0.0f);
                    if (abs(pos[1]) <= config.obstacle_scale)
                        particle_concentrations[idx] = 1.0f;
                    else
                        particle_concentrations[idx] = 0.0f;
                }
            }

            scene.particle_to_grid(particle_positions, particle_velocities, grid_velocity, grid_weight_sum);
            scene.particle_to_grid(particle_positions, particle_concentrations, grid_concentration, grid_weight_sum);
            scene.particle_to_grid(particle_positions, particle_temperatures, grid_temperature, grid_weight_sum);

            if (config.domain_type == DomainType::UnboundedDomain) {
                scene.grid_breadth_first_fill(grid_winding_number, grid_concentration);
                scene.grid_breadth_first_fill(grid_winding_number, grid_temperature);
            }

            utils::save_field(
                grid_velocity, config.output_dir + "/velocity_" + std::to_string(step) + ".vector", config.grid_res
            );
            utils::save_field(
                grid_concentration, config.output_dir + "/concentration_" + std::to_string(step) + ".scalar",
                config.grid_res
            );
            utils::save_field(
                grid_temperature, config.output_dir + "/temperature_" + std::to_string(step) + ".scalar",
                config.grid_res
            );

            auto stop_time = std::chrono::high_resolution_clock::now();

            std::cout << std::chrono::high_resolution_clock::now() << ": step " << step << ", t=" << step * config.dt
                      << " ("
                      << (double)std::chrono::duration_cast<std::chrono::milliseconds>(stop_time - start_time).count() /
                             1000.0
                      << "s)" << std::endl;
        }
    }
    std::cout << std::endl;

    return 0;
}