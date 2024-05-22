// Standard Advection-Projection method with optional diffusion.
// The projection and adveciton are comined into one step in this version.

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

    thrust::universal_vector<owl::vec_t<float, Dim>> temp_evaluation_point(config.num_evaluation_points);
    thrust::universal_vector<float> temp_winding_number(config.num_evaluation_points);
    thrust::universal_vector<owl::vec_t<float, Dim>> velocity_k1(config.num_evaluation_points);
    thrust::universal_vector<owl::vec_t<float, Dim>> velocity_k2(config.num_evaluation_points);
    thrust::universal_vector<owl::vec_t<float, Dim>> velocity_k3(config.num_evaluation_points);
    thrust::universal_vector<owl::vec_t<float, Dim>> velocity_k4(config.num_evaluation_points);

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
                case 3: // vortex
                {
                    const owl::vec_t<float, Dim> center1(-0.7, 1. / 6.);
                    const owl::vec_t<float, Dim> center2(-0.7, -1. / 6.);
                    const float radius = 0.8 * 1. / 6.;

                    owl::vec2f r1 = pos - center1;
                    if (owl::length2(r1) <= radius * radius) new_concentration_ptr[idx] = 1.0f;

                    owl::vec2f r2 = pos - center2;
                    if (owl::length2(r2) <= radius * radius) new_concentration_ptr[idx] = -1.0f;

                    // numerical intergation of biot-savart integral for two vorticities
                    // with magnitude 2pi in two circles.
                    owl::vec2f new_vel(0., 0.);
                    int r_div = 128;
                    int a_div = 128;
                    for (int _r = 0; _r < r_div; _r++) {
                        float r = (_r + 0.5f) * radius / r_div;
                        for (int _a = 0; _a < a_div; _a++) {
                            float a = M_PIf * 2 * _a / a_div;
                            owl::vec2f c1 = center1 + r * owl::vec2f(cos(a), sin(a));
                            owl::vec2f r1 = pos - c1;
                            new_vel += r * owl::vec2f(-r1.y, r1.x) / owl::length2(r1);

                            owl::vec2f c2 = center2 + r * owl::vec2f(cos(a), sin(a));
                            owl::vec2f r2 = pos - c2;
                            new_vel -= r * owl::vec2f(-r2.y, r2.x) / owl::length2(r2);
                        }
                    }
                    new_velocity_ptr[idx] = new_vel / (float)(r_div * a_div);
                    break;
                }
                case 4: // another vortex pair scene, but for the bunny boundary.
                {
                    const owl::vec_t<float, Dim> center1(-0.25f, -0.1f);
                    const owl::vec_t<float, Dim> center2(-0.1f, -0.25f);
                    const float radius = M_SQRT1_2f * 0.15f;

                    owl::vec2f r1 = pos - center1;
                    if (owl::length2(r1) <= radius * radius) new_concentration_ptr[idx] = -1.0f;

                    owl::vec2f r2 = pos - center2;
                    if (owl::length2(r2) <= radius * radius) new_concentration_ptr[idx] = 1.0f;

                    // numerical intergation of biot-savart integral for two vorticities
                    // with magnitude 2pi in two circles.
                    owl::vec2f new_vel(0., 0.);
                    int r_div = 128;
                    int a_div = 128;
                    for (int _r = 0; _r < r_div; _r++) {
                        float r = (_r + 0.5f) * radius / r_div;
                        for (int _a = 0; _a < a_div; _a++) {
                            float a = M_PIf * 2 * _a / a_div;
                            owl::vec2f c1 = center1 + r * owl::vec2f(cos(a), sin(a));
                            owl::vec2f r1 = pos - c1;
                            new_vel -= r * owl::vec2f(-r1.y, r1.x) / owl::length2(r1);

                            owl::vec2f c2 = center2 + r * owl::vec2f(cos(a), sin(a));
                            owl::vec2f r2 = pos - c2;
                            new_vel += r * owl::vec2f(-r2.y, r2.x) / owl::length2(r2);
                        }
                    }
                    new_velocity_ptr[idx] = new_vel / (float)(r_div * a_div);
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

    std::cout << "Starting time stepping... " << std::endl;

    // Step 0 is without advection and diffusion. We do this so the first diffusion step will still use an
    // incompressible velocity field.
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
    for (int step = 1; step <= config.time_steps; step++) {
        auto start_time = std::chrono::high_resolution_clock::now();

        thrust::swap(old_velocity, new_velocity);
        thrust::swap(old_concentration, new_concentration);
        thrust::swap(old_temperature, new_temperature);

        // diffuse
        if (enable_concentration_diffusion)
            scene.diffuse_scalar(
                evaluation_point, winding_number, new_concentration, random_state, winding_number, old_concentration,
                old_velocity, config.concentration_diffusion_coefficient, config.dt, false
            );
        else
            thrust::swap(new_concentration, old_concentration);

        if (enable_temperature_diffusion)
            scene.diffuse_scalar(
                evaluation_point, winding_number, new_temperature, random_state, winding_number, old_temperature,
                old_velocity, config.temperature_diffusion_coefficient, config.dt, false
            );
        else
            thrust::swap(new_temperature, old_temperature);

        if (enable_velocity_diffusion)
            scene.diffuse_velocity(
                evaluation_point, winding_number, new_velocity, random_state, winding_number, old_velocity,
                old_velocity, config.velocity_diffusion_coefficient, config.dt, false
            );
        else
            thrust::swap(new_velocity, old_velocity);

        // for buoyancy simulation, add concentration and temperature source and apply the acceleration
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

                const owl::vec_t<float, Dim> acceleration =
                    (config.buoyancy_alpha * new_concentration[idx] - config.buoyancy_beta * new_temperature[idx]) *
                    config.buoyancy_gravity;
                new_velocity[idx] += config.dt * acceleration;
            }
        }

        // for the karman vortex street scene, set the concentration and velocity at the inlet.
        if (config.scene_number == 2) {
            for (int idx = 0; idx < config.num_evaluation_points; idx++) {
                owl::vec_t<int, Dim> idx_md = utils::unflatten(idx, config.grid_res);
                owl::vec_t<float, Dim> pos = utils::idx_to_domain_point<Dim>(idx, config.grid_res, config.domain_size);
                if (idx_md[0] == 0) {
                    new_velocity[idx] = owl::vec_t<float, Dim>(1.0f, 0.0f);
                    if (abs(pos[1]) <= config.obstacle_scale)
                        new_concentration[idx] = 1.0f;
                    else
                        new_concentration[idx] = 0.0f;
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

        thrust::swap(old_velocity, new_velocity);
        thrust::swap(old_concentration, new_concentration);
        thrust::swap(old_temperature, new_temperature);

        // project and advect all three fields
        {
            // If we are using VPL, use the same cache data for all substeps in advection.
            if (config.enable_vpl)
                scene.project_vpl_construct(
                    vpl_data, vpl_random_state, winding_number, old_velocity, old_velocity, 0.0f, false,
                    config.velocity_sources
                );

            // Define this for convenience.
            auto project_temp_evaluation_points = [&](thrust::universal_vector<owl::vec_t<float, Dim>> &new_velocity) {
                scene.winding_number(temp_evaluation_point, temp_winding_number, random_state);
                if (config.enable_vpl)
                    scene.project_vpl_gather(
                        temp_evaluation_point, temp_winding_number, new_velocity, random_state, winding_number,
                        old_velocity, old_velocity, vpl_data, 0.0f, false, config.velocity_sources
                    );
                else
                    scene.project_pointwise(
                        temp_evaluation_point, temp_winding_number, new_velocity, random_state, winding_number,
                        old_velocity, old_velocity, 0.0f, false, config.velocity_sources
                    );
            };

            // We evaluate the velocity field at the original point first.
            temp_evaluation_point = evaluation_point;
            project_temp_evaluation_points(velocity_k1);

            switch (config.advection_mode) {
            case AdvectionMode::Euler: {
                for (int idx = 0; idx < config.num_evaluation_points; idx++)
                    temp_evaluation_point[idx] = evaluation_point[idx] - config.dt * velocity_k1[idx];
                break;
            }
            case AdvectionMode::MacCormack: {
                for (int idx = 0; idx < config.num_evaluation_points; idx++)
                    temp_evaluation_point[idx] = evaluation_point[idx] - config.dt * velocity_k1[idx];
                project_temp_evaluation_points(velocity_k2);

                for (int idx = 0; idx < config.num_evaluation_points; idx++)
                    temp_evaluation_point[idx] =
                        evaluation_point[idx] - 0.5f * config.dt * (velocity_k1[idx] + velocity_k2[idx]);
                break;
            }
            case AdvectionMode::RK3: {
                for (int idx = 0; idx < config.num_evaluation_points; idx++)
                    temp_evaluation_point[idx] = evaluation_point[idx] - 0.5f * config.dt * velocity_k1[idx];
                project_temp_evaluation_points(velocity_k2);

                for (int idx = 0; idx < config.num_evaluation_points; idx++)
                    temp_evaluation_point[idx] = evaluation_point[idx] - 0.75f * config.dt * velocity_k2[idx];
                project_temp_evaluation_points(velocity_k3);

                for (int idx = 0; idx < config.num_evaluation_points; idx++)
                    temp_evaluation_point[idx] =
                        evaluation_point[idx] -
                        config.dt * (2.f * velocity_k1[idx] + 3.f * velocity_k2[idx] + 4.f * velocity_k3[idx]) / 9.f;
                break;
            }
            case AdvectionMode::RK4: {
                for (int idx = 0; idx < config.num_evaluation_points; idx++)
                    temp_evaluation_point[idx] = evaluation_point[idx] - 0.5f * config.dt * velocity_k1[idx];
                project_temp_evaluation_points(velocity_k2);

                for (int idx = 0; idx < config.num_evaluation_points; idx++)
                    temp_evaluation_point[idx] = evaluation_point[idx] - 0.5f * config.dt * velocity_k2[idx];
                project_temp_evaluation_points(velocity_k3);

                for (int idx = 0; idx < config.num_evaluation_points; idx++)
                    temp_evaluation_point[idx] = evaluation_point[idx] - config.dt * velocity_k3[idx];
                project_temp_evaluation_points(velocity_k4);

                for (int idx = 0; idx < config.num_evaluation_points; idx++)
                    temp_evaluation_point[idx] =
                        evaluation_point[idx] -
                        config.dt *
                            (velocity_k1[idx] + 2.f * velocity_k2[idx] + 2.f * velocity_k3[idx] + velocity_k4[idx]) /
                            6.f;

                break;
            }
            }

            {
                // evaluate the velocity field at the backward-advected point
                project_temp_evaluation_points(new_velocity);

                // advect concentration and temperature
                for (int idx = 0; idx < config.num_evaluation_points; idx++) {
                    new_concentration[idx] = utils::sample_buffer_linear(
                        utils::domain_point_to_idx(temp_evaluation_point[idx], config.grid_res, config.domain_size),
                        old_concentration.data().get(), config.grid_res, utils::WrapMode::ClampToEdge
                    );
                    new_temperature[idx] = utils::sample_buffer_linear(
                        utils::domain_point_to_idx(temp_evaluation_point[idx], config.grid_res, config.domain_size),
                        old_temperature.data().get(), config.grid_res, utils::WrapMode::ClampToEdge
                    );
                }
            }
        }

        if (config.domain_type == DomainType::UnboundedDomain) {
            scene.grid_breadth_first_fill(winding_number, new_concentration);
            scene.grid_breadth_first_fill(winding_number, new_temperature);
        }

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