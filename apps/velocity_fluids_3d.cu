// Standard Advection-Projection method with optional diffusion.
// The advection and the next step can be optionally done in place with enable_inplace_advection.

#include "common.cuh"
#include "owl/common/math/vec.h"
#include "utils.hpp"
#include "velocity_fluids_host.cuh"
#include <cmath>
#include <filesystem>

constexpr int Dim = 3;

int main(int argc, char *argv[]) {
    const char *input_json_file = argv[1];
    Config<Dim> config = load_config<Dim>(input_json_file);

    std::filesystem::create_directories(config.output_dir);
    std::filesystem::copy(
        input_json_file, config.output_dir + "/config.json", std::filesystem::copy_options::overwrite_existing
    );

    Scene<Dim> scene(config);

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

    // Load smoke source field from file
    std::vector<float> smoke_source;
    if (config.scene_number == 1) {
        std::cout << "smoke source:\t" << config.json_config["smoke_source_field"].get<std::string>() << std::endl;
        utils::load_field<2, float>(smoke_source, config.json_config["smoke_source_field"].get<std::string>());
    }

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
        thrust::for_each(
            thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(config.num_evaluation_points),
            [=] __device__(int idx) {
                owl::vec_t<float, Dim> pos = utils::idx_to_domain_point<Dim>(idx, grid_res, domain_size);
                new_velocity_ptr[idx] = owl::vec_t<float, Dim>(0.0, 0.0, 0.0);
                new_concentration_ptr[idx] = 0.0;
                new_temperature_ptr[idx] = 0.0;

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
    bool enable_velocity_diffusion = config.velocity_diffusion_coefficient > 0;
    bool enable_projection_inplace_advection = config.enable_inplace_advection && !enable_velocity_diffusion;
    for (int step = 1; step <= config.time_steps; step++) {
        auto start_time = std::chrono::high_resolution_clock::now();

        thrust::swap(old_velocity, new_velocity);
        thrust::swap(old_concentration, new_concentration);
        thrust::swap(old_temperature, new_temperature);

        // move the solid obstacles
        if (config.solid_velocity != utils::zero<owl::vec_t<float, Dim>>()) {
            scene.setTranslation(config.obstacle_shift + step * config.dt * config.solid_velocity);
            scene.winding_number(evaluation_point, winding_number, random_state);
            utils::save_field(
                winding_number, config.output_dir + "/winding_number_" + std::to_string(step) + ".scalar",
                config.grid_res
            );
        }

        // advect.
        // The result of velocity advection may not be used in the computation if enable_inplace_advection is true.
        scene.advect(evaluation_point, old_velocity, old_concentration, new_concentration, config.dt);
        scene.advect(evaluation_point, old_velocity, old_temperature, new_temperature, config.dt);
        scene.advect(evaluation_point, old_velocity, old_velocity, new_velocity, config.dt);

        // buoyancy
        if (config.buoyancy_alpha != 0.0 || config.buoyancy_beta != 0.0) {
            for (int idx = 0; idx < config.num_evaluation_points; idx++) {
                // These are hardcoded.
                owl::vec_t<float, Dim> pos = utils::idx_to_domain_point<Dim>(idx, config.grid_res, config.domain_size);

                if (config.scene_number == 1) {
                    owl::vec3i idx_3d = utils::unflatten(idx, config.grid_res);
                    owl::vec2i idx_2d = owl::vec2i(idx_3d.x, idx_3d.y);
                    int idx_2d_flat = utils::flatten(idx_2d, owl::vec2i(config.grid_res.x, config.grid_res.y));

                    bool inside_smoke_source = abs(pos.z <= 0.25f) && smoke_source[idx_2d_flat] > 0.5f;

                    if (inside_smoke_source) {
                        new_concentration[idx] =
                            min(new_concentration[idx] + config.dt * config.concentration_rate, 1.0f);
                        new_temperature[idx] +=
                            (1.0 - std::exp(-config.temperature_rate * config.dt)) * (1.0 - new_temperature[idx]);
                    }
                } else if (config.scene_number == 0) { // add concentration and temperature box source
                    if (abs(pos.x) < 0.125f && abs(pos.y - -1.25f) < 0.125f && abs(pos.z) < 0.125f) {
                        new_concentration[idx] =
                            min(new_concentration[idx] + config.dt * config.concentration_rate, 1.0f);
                        new_temperature[idx] +=
                            (1.0 - std::exp(-config.temperature_rate * config.dt)) * (1.0 - new_temperature[idx]);
                    }
                }

                const owl::vec_t<float, Dim> acceleration =
                    (config.buoyancy_alpha * new_concentration[idx] - config.buoyancy_beta * new_temperature[idx]) *
                    config.buoyancy_gravity;
                new_velocity[idx] += config.dt * acceleration;
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
        if (enable_velocity_diffusion) {
            utils::save_field(
                new_velocity, config.output_dir + "/velocity_prediffuse_" + std::to_string(step) + ".vector",
                config.grid_res
            );
            if (!config.enable_inplace_advection) thrust::swap(old_velocity, new_velocity);
            scene.diffuse_velocity(
                evaluation_point, winding_number, new_velocity, random_state, winding_number, old_velocity,
                old_velocity, config.velocity_diffusion_coefficient, config.dt, config.enable_inplace_advection
            );
        }

        if (config.domain_type == DomainType::UnboundedDomain) {
            scene.grid_breadth_first_fill(winding_number, new_concentration);
            scene.grid_breadth_first_fill(winding_number, new_temperature);
        }

        utils::save_field(
            new_velocity, config.output_dir + "/velocity_preproject_" + std::to_string(step) + ".vector",
            config.grid_res
        );
        utils::save_field(
            new_concentration, config.output_dir + "/concentration_" + std::to_string(step) + ".scalar", config.grid_res
        );
        utils::save_field(
            new_temperature, config.output_dir + "/temperature_" + std::to_string(step) + ".scalar", config.grid_res
        );

        if (!enable_projection_inplace_advection) thrust::swap(old_velocity, new_velocity);

        // project
        scene.project(
            evaluation_point, winding_number, new_velocity, random_state, winding_number, old_velocity, old_velocity,
            vpl_data, vpl_random_state, config.dt, enable_projection_inplace_advection, config.velocity_sources
        );

        utils::save_field(
            new_velocity, config.output_dir + "/velocity_" + std::to_string(step) + ".vector", config.grid_res
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