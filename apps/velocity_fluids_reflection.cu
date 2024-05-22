// Second-Order Advection-Reflection method.
// The advection and the projection in each substep can be optionally done in place with enable_inplace_advection.

#include "utils.hpp"
#include "velocity_fluids_host.cuh"
#include <filesystem>

constexpr int Dim = 2;

int main(int argc, char *argv[]) {
    const char *input_json_file = argv[1];
    Config<Dim> config = load_config<Dim>(input_json_file);

    std::filesystem::create_directories(config.output_dir);
    std::filesystem::copy(
        input_json_file, config.output_dir + "/config.json", std::filesystem::copy_options::overwrite_existing
    );

    Scene<Dim> scene(config);

    bool enable_velocity_diffusion = config.velocity_diffusion_coefficient > 0;
    if (enable_velocity_diffusion) {
        std::cerr << "velocity diffusion is not supported for adveciton-refleciton." << std::endl;
        return 1;
    }

    if ((config.buoyancy_alpha != 0.0 || config.buoyancy_beta != 0.0) && config.enable_inplace_advection) {
        std::cerr << "buoyancy force is not supported for inplace advection." << std::endl;
        return 1;
    }

    std::cout << "Initializing buffers..." << std::endl;
    thrust::universal_vector<utils::randState_t> random_state(config.num_evaluation_points);
    utils::random_states_init(random_state);

    thrust::universal_vector<owl::vec_t<float, Dim>> evaluation_point(config.num_evaluation_points);
    scene.initialize_evaluation_points(evaluation_point);

    thrust::universal_vector<float> winding_number(config.num_evaluation_points);
    thrust::universal_vector<owl::vec_t<float, Dim>> old_velocity(config.num_evaluation_points);
    thrust::universal_vector<owl::vec_t<float, Dim>> new_velocity(config.num_evaluation_points);
    thrust::universal_vector<float> old_concentration(config.num_evaluation_points);
    thrust::universal_vector<float> new_concentration(config.num_evaluation_points);
    thrust::universal_vector<float> old_temperature(config.num_evaluation_points);
    thrust::universal_vector<float> new_temperature(config.num_evaluation_points);

    thrust::universal_vector<ProjectionVPLRecord<Dim>> vpl_data;
    thrust::universal_vector<utils::randState_t> vpl_random_state;

    if (config.enable_vpl) {
        vpl_data.resize(config.num_path_samples * (config.path_length + 1));
        vpl_random_state.resize(config.num_path_samples);
        utils::random_states_init(vpl_random_state);
    }

    // Set initial conditions
    {
        scene.winding_number(evaluation_point, winding_number, random_state);
        owl::vec_t<float, Dim> *new_velocity_ptr = new_velocity.data().get();
        float *new_concentration_ptr = new_concentration.data().get();
        float *new_temperature_ptr = new_temperature.data().get();
        const float *winding_number_ptr = winding_number.data().get();
        owl::vec_t<int, Dim> grid_res = config.grid_res;
        owl::vec_t<float, Dim> domain_size = config.domain_size;
        DomainType domain_type = config.domain_type;
        owl::vec_t<float, Dim> solid_velocity = config.solid_velocity;
        int scene_number = config.scene_number;
        float obstacle_scale = config.obstacle_scale;
        thrust::for_each(
            thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(config.num_evaluation_points),
            [=] __device__(int idx) {
                owl::vec_t<float, Dim> pos = utils::idx_to_domain_point<Dim>(idx, grid_res, domain_size);
                new_velocity_ptr[idx] = owl::vec_t<float, Dim>(0.0, 0.0);
                new_concentration_ptr[idx] = 0.0;
                new_temperature_ptr[idx] = 0.0;

                switch (scene_number) {
                case 0: // zero initial conditions
                    break;
                case 1: // circulating velocity
                {
                    float r = owl::length(pos);
                    float magnitude = max(min(r, 0.95f - r), 0.f); // r > 0.95f ? 0.0f : r;
                    owl::vec_t<float, Dim> direction = owl::normalize(owl::vec_t<float, Dim>(pos.y, -pos.x));
                    new_velocity_ptr[idx] = magnitude * direction;
                    break;
                }
                case 2: // karman vortex street
                {
                    owl::vec_t<int, Dim> idx_md = utils::unflatten(idx, grid_res);
                    new_velocity_ptr[idx] = owl::vec_t<float, Dim>(1.f, 0.f);
                    if (idx_md[0] == 0 && abs(pos[1]) <= obstacle_scale)
                        new_concentration_ptr[idx] = 1.f;
                    else
                        new_concentration_ptr[idx] = 0.f;
                    break;
                }
                default: break;
                }

                // Point inside the solid obstacles should have the solid velocity.
                float blend_weight =
                    domain_type == DomainType::BoundedDomain ? winding_number_ptr[idx] : 1.0f + winding_number_ptr[idx];
                new_velocity_ptr[idx] = blend_weight * new_velocity_ptr[idx] + (1.0f - blend_weight) * solid_velocity;
            }
        );
    }

    utils::save_field(winding_number, config.output_dir + "/winding_number_0.scalar", config.grid_res);
    utils::save_field(new_velocity, config.output_dir + "/velocity_preproject_0.vector", config.grid_res);
    utils::save_field(new_concentration, config.output_dir + "/concentration_0.scalar", config.grid_res);
    utils::save_field(new_temperature, config.output_dir + "/temperature_0.scalar", config.grid_res);

    thrust::universal_vector<owl::vec_t<float, Dim>> half_velocity(config.num_evaluation_points);
    thrust::universal_vector<owl::vec_t<float, Dim>> half_velocity_force(config.num_evaluation_points);
    thrust::universal_vector<owl::vec_t<float, Dim>> half_velocity_hat(config.num_evaluation_points);
    thrust::universal_vector<owl::vec_t<float, Dim>> half_velocity_hat_adv(config.num_evaluation_points);

    std::cout << "Starting time stepping... " << std::endl;
    // Step 0 is without advection and diffusion.
    {
        auto start_time = std::chrono::high_resolution_clock::now();
        thrust::swap(old_velocity, new_velocity);
        scene.project(
            evaluation_point, winding_number, new_velocity, random_state, winding_number, old_velocity, old_velocity,
            vpl_data, vpl_random_state, config.dt, false, config.velocity_sources
        );
        utils::save_field(new_velocity, config.output_dir + "/velocity_0.vector", config.grid_res);
        auto stop_time = std::chrono::high_resolution_clock::now();

        std::cout << std::chrono::high_resolution_clock::now() << ": step 0, t=0 ("
                  << (double)std::chrono::duration_cast<std::chrono::milliseconds>(stop_time - start_time).count() /
                         1000.0
                  << "s)" << std::endl;
    }

    bool enable_concentration_diffusion = config.concentration_diffusion_coefficient > 0;
    bool enable_temperature_diffusion = config.temperature_diffusion_coefficient > 0;
    for (int step = 1; step <= config.time_steps; step++) {
        auto start_time = std::chrono::high_resolution_clock::now();

        thrust::swap(old_velocity, new_velocity);
        thrust::swap(old_concentration, new_concentration);
        thrust::swap(old_temperature, new_temperature);

        // for the karman vortex street scene, set the concentration and velocity at the inlet.
        if (config.scene_number == 2) {
            for (int idx = 0; idx < config.num_evaluation_points; idx++) {
                owl::vec_t<int, Dim> idx_md = utils::unflatten(idx, config.grid_res);
                owl::vec_t<float, Dim> pos = utils::idx_to_domain_point<Dim>(idx, config.grid_res, config.domain_size);
                if (idx_md[0] == 0) {
                    old_velocity[idx] = owl::vec_t<float, Dim>(1.0f, 0.0f);
                    if (abs(pos[1]) <= config.obstacle_scale)
                        old_concentration[idx] = 1.0f;
                    else
                        old_concentration[idx] = 0.0f;
                }
            }
        }

        // move the solid obstacles
        if (config.solid_velocity != utils::zero<owl::vec_t<float, Dim>>()) {
            scene.setTranslation(config.obstacle_shift + step * config.dt * config.solid_velocity);
            scene.winding_number(evaluation_point, winding_number, random_state);
            utils::save_field(
                winding_number, config.output_dir + "/winding_number_" + std::to_string(step) + ".scalar",
                config.grid_res
            );
        }

        // add concentration and temperature box source
        if (config.buoyancy_alpha != 0.0 || config.buoyancy_beta != 0.0) {
            for (int idx = 0; idx < config.num_evaluation_points; idx++) {
                // These are hardcoded.
                owl::vec_t<float, Dim> pos = utils::idx_to_domain_point<Dim>(idx, config.grid_res, config.domain_size);

                // add concentration and temperature box source
                if (abs(pos.x - 0.0625f) < 0.125f && abs(pos.y - -0.75f) < 0.125f) {
                    new_concentration[idx] = min(new_concentration[idx] + config.dt * config.concentration_rate, 1.0f);
                    new_temperature[idx] +=
                        (1.0 - std::exp(-config.temperature_rate * config.dt)) * (1.0 - new_temperature[idx]);
                }
            }
        }

        // diffuse
        if (enable_concentration_diffusion) {
            utils::save_field(
                new_concentration, config.output_dir + "/concentration_prediffuse_" + std::to_string(step) + ".scalar",
                config.grid_res
            );
            if (!config.enable_inplace_advection) thrust::swap(old_concentration, new_concentration);
            scene.diffuse_scalar(
                evaluation_point, winding_number, new_concentration, random_state, winding_number, old_concentration,
                old_velocity, config.concentration_diffusion_coefficient, config.dt, config.enable_inplace_advection
            );
        }
        if (enable_temperature_diffusion) {
            utils::save_field(
                new_temperature, config.output_dir + "/temperature_prediffuse_" + std::to_string(step) + ".scalar",
                config.grid_res
            );
            if (!config.enable_inplace_advection) thrust::swap(old_temperature, new_temperature);
            scene.diffuse_scalar(
                evaluation_point, winding_number, new_temperature, random_state, winding_number, old_temperature,
                old_velocity, config.temperature_diffusion_coefficient, config.dt, config.enable_inplace_advection
            );
        }

        // advect velocity by half step. This advection is necessary even if inplace advection is enabled for the
        // computation of  u_hat^{1/2}.
        scene.advect(evaluation_point, old_velocity, old_velocity, half_velocity, config.dt / 2.f);
        // now, half_velocity stores u_a^0

        scene.advect(evaluation_point, old_velocity, old_concentration, new_concentration, config.dt / 2.f);
        scene.advect(evaluation_point, old_velocity, old_temperature, new_temperature, config.dt / 2.f);

        if (config.domain_type == DomainType::UnboundedDomain) {
            scene.grid_breadth_first_fill(winding_number, new_concentration);
            scene.grid_breadth_first_fill(winding_number, new_temperature);
        }

        // buoyancy acceleration force
        if (config.buoyancy_alpha != 0.0 || config.buoyancy_beta != 0.0) {
            for (int idx = 0; idx < config.num_evaluation_points; idx++) {
                const owl::vec_t<float, Dim> acceleration =
                    (config.buoyancy_alpha * new_concentration[idx] - config.buoyancy_beta * new_temperature[idx]) *
                    config.buoyancy_gravity;
                half_velocity_force[idx] = half_velocity[idx] + config.dt * acceleration;
            }
        } else
            half_velocity_force = half_velocity;

        // first advection and projection combined
        scene.project(
            evaluation_point, winding_number, new_velocity, random_state, winding_number,
            config.enable_inplace_advection ? old_velocity : half_velocity_force, old_velocity, vpl_data,
            vpl_random_state, config.dt / 2.f, config.enable_inplace_advection, config.velocity_sources
        );
        // now, new_velocity stores u^{1/2}

        {
            owl::vec_t<float, Dim> *half_velocity_ptr = half_velocity.data().get();
            owl::vec_t<float, Dim> *half_velocity_hat_ptr = half_velocity_hat.data().get();
            owl::vec_t<float, Dim> *half_velocity_hat_adv_ptr = half_velocity_hat_adv.data().get();
            owl::vec_t<float, Dim> *new_velocity_ptr = new_velocity.data().get();
            owl::vec_t<float, Dim> *old_velocity_ptr = old_velocity.data().get();
            owl::vec_t<int, Dim> grid_res = config.grid_res;
            thrust::for_each(
                thrust::make_counting_iterator<int>(0),
                thrust::make_counting_iterator<int>(config.num_evaluation_points),
                [=] __device__(int idx) {
                    owl::vec_t<int, Dim> idx_2d = utils::unflatten(idx, grid_res);
                    half_velocity_hat_ptr[idx] = 2.f * new_velocity_ptr[idx] - half_velocity_ptr[idx];
                    half_velocity_hat_adv_ptr[idx] = 2.f * new_velocity_ptr[idx] - old_velocity_ptr[idx];
                }
            );

            if (config.scene_number == 2)
                thrust::for_each(
                    thrust::make_counting_iterator<int>(0),
                    thrust::make_counting_iterator<int>(config.num_evaluation_points),
                    [=] __device__(int idx) {
                        owl::vec_t<int, Dim> idx_2d = utils::unflatten(idx, grid_res);
                        if (idx_2d[0] == 0) {
                            half_velocity_hat_ptr[idx] = owl::vec_t<float, Dim>(1.f, 0.0f);
                            half_velocity_hat_adv_ptr[idx] = owl::vec_t<float, Dim>(1.f, 0.0f);
                        }
                    }
                );
        }
        // now, half_velocity_hat stores u_hat^{1/2}
        // half_velocity_hat_adv stores 2u^{1/2} - u^{0}

        // advect velocity by half step.
        if (!config.enable_inplace_advection)
            scene.advect(evaluation_point, half_velocity_hat_adv, half_velocity_hat, half_velocity, config.dt / 2.f);
        // now, half_velocity stores u_a^{1/2}

        thrust::swap(new_concentration, old_concentration);
        thrust::swap(new_temperature, old_temperature);
        scene.advect(evaluation_point, half_velocity_hat_adv, old_concentration, new_concentration, config.dt / 2.f);
        scene.advect(evaluation_point, half_velocity_hat_adv, old_temperature, new_temperature, config.dt / 2.f);
        if (config.domain_type == DomainType::UnboundedDomain) {
            scene.grid_breadth_first_fill(winding_number, new_concentration);
            scene.grid_breadth_first_fill(winding_number, new_temperature);
        }

        scene.project(
            evaluation_point, winding_number, new_velocity, random_state, winding_number,
            config.enable_inplace_advection ? half_velocity_hat : half_velocity, half_velocity_hat_adv, vpl_data,
            vpl_random_state, config.dt / 2.f, config.enable_inplace_advection, config.velocity_sources
        );
        // now, new_velocity stores u^1

        utils::save_field(
            new_velocity, config.output_dir + "/velocity_" + std::to_string(step) + ".vector", config.grid_res
        );
        utils::save_field(
            new_concentration, config.output_dir + "/concentration_" + std::to_string(step) + ".scalar", config.grid_res
        );
        utils::save_field(
            new_temperature, config.output_dir + "/temperature_" + std::to_string(step) + ".scalar", config.grid_res
        );

        auto stop_time = std::chrono::high_resolution_clock::now();

        std::cout << std::chrono::high_resolution_clock::now() << ": step " << step << ", t=" << step * config.dt
                  << " ("
                  << (double)std::chrono::duration_cast<std::chrono::milliseconds>(stop_time - start_time).count() /
                         1000.0
                  << "s)" << std::endl;
    }
    std::cout << std::endl;

    return 0;
}