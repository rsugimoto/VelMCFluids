/* The core implementation of Neumann WoB for velocity-based fluids. */
#include "owl/owl_host.h"
#include "velocity_fluids_host.cuh"

#include <iostream>
#include <string>

#include <thrust/device_vector.h>

#include "owl/common/math/AffineSpace.h"
#include "owl/common/math/vec.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#include "utils.hpp"
#include "velocity_fluids.cuh"

// #define VELMCFLUIDS_VERBOSE

#ifdef VELMCFLUIDS_VERBOSE
#define VELMCFLUIDS_TIME_BEGIN auto __start_time = std::chrono::high_resolution_clock::now();

#define VELMCFLUIDS_TIME_END                                                                                           \
    auto __end_time = std::chrono::high_resolution_clock::now();                                                       \
    std::cout << __func__ << " "                                                                                       \
              << (double)std::chrono::duration_cast<std::chrono::milliseconds>(__end_time - __start_time).count() /    \
                     1000.0                                                                                            \
              << " s" << std::endl;
#else
#define VELMCFLUIDS_TIME_BEGIN
#define VELMCFLUIDS_TIME_END
#endif

extern "C" char geometry_optix_ptx[];
extern "C" char velocity_fluids_optix_ptx[];
extern "C" char fluids_common_optix_ptx[];

template <int Dim> Config<Dim> load_config(const char *file) {
    Config<Dim> config = {0};

    using json = nlohmann::json;
    json &json_config = config.json_config;
    {
        std::ifstream config_file(file);
        if (!config_file) {
            std::cerr << "Failed to load config file: " << file << std::endl;
            exit(1);
        }
        config_file >> json_config;
        config_file.close();
    }

    config.output_dir = json_config["output_dir"].get<std::string>();
    config.domain_type = stringToDomainType(json_config["domain_type"].get<std::string>().c_str());
    if (json_config.contains("obj_file")) {
        config.obj_file = json_config["obj_file"].get<std::string>();
        if (json_config.contains("invert_normals"))
            config.invert_normals = json_config["invert_normals"].get<bool>();
        else
            config.invert_normals = false;
        config.path_length = json_config["path_length"].get<int>();
        config.num_path_samples = json_config["num_path_samples"].get<int>();
        config.num_volume_samples_indirect = json_config["num_volume_samples_indirect"].get<int>();
        if (config.domain_type == DomainType::UnboundedDomain)
            config.num_pseudo_boundary_samples_indirect =
                json_config["num_pseudo_boundary_samples_indirect"].get<int>();
        config.num_winding_samples = json_config["num_winding_samples"].get<int>();
        config.num_init_winding_samples = json_config["num_init_winding_samples"].get<int>();
    }
    config.num_volume_samples_direct = json_config["num_volume_samples_direct"].get<int>();
    if (config.domain_type == DomainType::UnboundedDomain)
        config.num_pseudo_boundary_samples_direct = json_config["num_pseudo_boundary_samples_direct"].get<int>();
    if (json_config["grid_res"].is_array()) {
        if (json_config["grid_res"].size() != Dim) {
            std::cerr << "grid_res must have " << Dim << " elements." << std::endl;
            exit(1);
        }
        for (int i = 0; i < Dim; i++) config.grid_res[i] = json_config["grid_res"][i].get<int>();
    } else
        config.grid_res = json_config["grid_res"].get<int>();
    if (json_config["domain_size"].is_array()) {
        if (json_config["domain_size"].size() != Dim) {
            std::cerr << "domain_size must have " << Dim << " elements." << std::endl;
            exit(1);
        }
        for (int i = 0; i < Dim; i++) config.domain_size[i] = json_config["domain_size"][i].get<float>();
    } else
        config.domain_size = json_config["domain_size"].get<float>();
    config.time_steps = json_config["time_steps"].get<int>();
    config.dt = json_config["dt"].get<float>();
    config.dGdx_regularization = json_config["dGdx_regularization"].get<float>();
    config.num_evaluation_points = Dim == 2 ? config.grid_res[0] * config.grid_res[1]
                                            : config.grid_res[0] * config.grid_res[1] * config.grid_res[2];

    if (json_config.contains("solid_velocity")) {
        if (Dim == 2)
            config.solid_velocity = owl::cast_dim<Dim>(owl::vec_t<float, 2>(
                json_config["solid_velocity"][0].get<float>(), json_config["solid_velocity"][1].get<float>()
            ));
        else if (Dim == 3)
            config.solid_velocity = owl::cast_dim<Dim>(owl::vec_t<float, 3>(
                json_config["solid_velocity"][0].get<float>(), json_config["solid_velocity"][1].get<float>(),
                json_config["solid_velocity"][2].get<float>()
            ));
    } else
        config.solid_velocity = utils::zero<owl::vec_t<float, Dim>>();

    config.advection_mode = stringToAdvectionMode(json_config["advection_mode"].get<std::string>().c_str());
    config.enable_inplace_advection = json_config["enable_inplace_advection"].get<bool>();
    config.velocity_diffusion_coefficient = json_config.contains("velocity_diffusion_coefficient")
                                                ? json_config["velocity_diffusion_coefficient"].get<float>()
                                                : 0.0f;
    config.concentration_diffusion_coefficient = json_config.contains("concentration_diffusion_coefficient")
                                                     ? json_config["concentration_diffusion_coefficient"].get<float>()
                                                     : 0.0f;
    config.temperature_diffusion_coefficient = json_config.contains("temperature_diffusion_coefficient")
                                                   ? json_config["temperature_diffusion_coefficient"].get<float>()
                                                   : 0.0f;

    config.buoyancy_alpha = json_config.contains("buoyancy_alpha") ? json_config["buoyancy_alpha"].get<float>() : 0.0f;
    config.buoyancy_beta = json_config.contains("buoyancy_beta") ? json_config["buoyancy_beta"].get<float>() : 0.0f;
    if (json_config.contains("buoyancy_gravity")) {
        if (Dim == 2)
            config.buoyancy_gravity = owl::cast_dim<Dim>(owl::vec_t<float, 2>(
                json_config["buoyancy_gravity"][0].get<float>(), json_config["buoyancy_gravity"][1].get<float>()
            ));
        else if (Dim == 3)
            config.buoyancy_gravity = owl::cast_dim<Dim>(owl::vec_t<float, 3>(
                json_config["buoyancy_gravity"][0].get<float>(), json_config["buoyancy_gravity"][1].get<float>(),
                json_config["buoyancy_gravity"][2].get<float>()
            ));
    } else
        config.buoyancy_gravity = utils::zero<owl::vec_t<float, Dim>>();
    config.concentration_rate =
        json_config.contains("concentration_rate") ? json_config["concentration_rate"].get<float>() : 0.0f;
    config.temperature_rate =
        json_config.contains("temperature_rate") ? json_config["temperature_rate"].get<float>() : 0.0f;

    if (json_config.contains("obstacle_shift"))
        for (int i = 0; i < Dim; i++) config.obstacle_shift[i] = json_config["obstacle_shift"][i].get<float>();
    else
        config.obstacle_shift = utils::zero<owl::vec_t<float, Dim>>();

    config.obstacle_scale = json_config.contains("obstacle_scale") ? json_config["obstacle_scale"].get<float>() : 1.0f;
    if (config.enable_inplace_advection && (config.buoyancy_alpha > 0.0 || config.buoyancy_beta > 0.0)) {
        std::cerr << "Buoyancy force is not supported with inplace advection." << std::endl;
        exit(1);
    }
    if (json_config.contains("velocity_sources")) {
        for (auto &v : json_config["velocity_sources"]) {
            if (Dim == 2)
                config.velocity_sources.push_back(owl::cast_dim<Dim + 1>(
                    owl::vec_t<float, 3>(v[0].get<float>(), v[1].get<float>(), v[2].get<float>())
                ));
            else if (Dim == 3)
                config.velocity_sources.push_back(owl::cast_dim<Dim + 1>(
                    owl::vec_t<float, 4>(v[0].get<float>(), v[1].get<float>(), v[2].get<float>(), v[3].get<float>())
                ));
        }
    }

    config.enable_vpl = json_config["enable_vpl"].get<bool>();
    config.scene_number = json_config["scene_number"].get<int>();
    if (json_config.contains("enable_antithetic_sampling"))
        config.enable_antithetic_sampling = json_config["enable_antithetic_sampling"].get<bool>();
    else
        config.enable_antithetic_sampling = false;

    if (config.enable_antithetic_sampling &&
        (config.num_volume_samples_direct % 2 != 0 || config.num_volume_samples_indirect % 2 != 0))
        std::cout << "Warning: Antithetic sampling is enabled, but the number of samples is not a multiple of 2."
                  << std::endl;

    std::cout << "------------------------------------------------------------" << std::endl;
    std::cout << "begin time stamp\t: " << std::chrono::high_resolution_clock::now() << std::endl;
    std::cout << "build time stamp\t: " << __DATE__ << " " << __TIME__ << std::endl;
    std::cout << "input json file\t: " << file << std::endl;
    std::cout << "output directory\t: " << config.output_dir << std::endl;
    std::cout << "obj file\t: " << config.obj_file << std::endl;
    std::cout << "invert normals\t: " << config.invert_normals << std::endl;
    std::cout << "domain type\t:" << domainTypeToString(config.domain_type) << std::endl;
    std::cout << "#path length\t: " << config.path_length << std::endl;
    std::cout << "#path samples\t: " << config.num_path_samples << std::endl;
    std::cout << "#volume samples indirect\t: " << config.num_volume_samples_indirect << std::endl;
    std::cout << "#pseudo boundary samples indirect\t: " << config.num_pseudo_boundary_samples_indirect << std::endl;
    std::cout << "#volume samples direct\t: " << config.num_volume_samples_direct << std::endl;
    std::cout << "#pseudo boundary samples direct\t: " << config.num_pseudo_boundary_samples_direct << std::endl;
    std::cout << "#winding samples\t: " << config.num_winding_samples << std::endl;
    std::cout << "#init winding samples: \t" << config.num_init_winding_samples << std::endl;
    std::cout << "grid resolution\t: " << config.grid_res << std::endl;
    std::cout << "domain size\t: " << config.domain_size << std::endl;
    std::cout << "time steps\t: " << config.time_steps << std::endl;
    std::cout << "dt\t:" << config.dt << std::endl;
    std::cout << "dGdx regularization\t:" << config.dGdx_regularization << std::endl;
    std::cout << "solid velocity\t:" << config.solid_velocity << std::endl;
    std::cout << "advection mode\t: " << advectionModeToString(config.advection_mode) << std::endl;
    std::cout << "enable inplace advection\t:" << config.enable_inplace_advection << std::endl;
    std::cout << "enable vpl\t:" << config.enable_vpl << std::endl;
    std::cout << "enable antithetic sampling\t:" << config.enable_antithetic_sampling << std::endl;
    std::cout << "velocity diffusion coefficient\t:" << config.velocity_diffusion_coefficient << std::endl;
    std::cout << "concentration diffusion coefficient\t:" << config.concentration_diffusion_coefficient << std::endl;
    std::cout << "temperature diffusion coefficient\t:" << config.temperature_diffusion_coefficient << std::endl;
    std::cout << "buoyancy alpha\t:" << config.buoyancy_alpha << std::endl;
    std::cout << "buoyancy beta\t:" << config.buoyancy_beta << std::endl;
    std::cout << "buoyancy gravity\t:" << config.buoyancy_gravity << std::endl;
    std::cout << "concentration rate\t:" << config.concentration_rate << std::endl;
    std::cout << "temperature rate\t:" << config.temperature_rate << std::endl;
    std::cout << "obstacle shift\t:" << config.obstacle_shift << std::endl;
    std::cout << "obstacle scale\t:" << config.obstacle_scale << std::endl;
    std::cout << "#velocity sources\t:" << config.velocity_sources.size() << std::endl;
    for (auto &v : config.velocity_sources) std::cout << "\t" << v << std::endl;
    std::cout << "scene number\t:" << config.scene_number << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;

    return config;
}

template Config<2> load_config<2>(const char *file);
template Config<3> load_config<3>(const char *file);

template <int Dim> Scene<Dim>::Scene(const Config<Dim> &config) : config(config) {
    std::cout << "Loading scene config..." << std::endl;

    /* Load scene data to CPU memory */
    const std::string &obj_file = config.obj_file;
    static_assert(std::is_same_v<tinyobj::real_t, float>, "tinyobj::real_t must be float");

    tinyobj::ObjReaderConfig reader_config;
    tinyobj::ObjReader reader;
    if (!obj_file.empty()) {
        if (!reader.ParseFromFile(obj_file, reader_config)) {
            if (!reader.Error().empty()) { std::cerr << "TinyObjReader: " << reader.Error(); }
            exit(1);
        }
        if (!reader.Warning().empty()) { std::cout << "TinyObjReader: " << reader.Warning(); }
    }
    auto &attrib = reader.GetAttrib();
    auto &shapes = reader.GetShapes();

    int num_primitives;
    std::vector<owl::vec3f> vertices;
    std::vector<owl::vec3i> indices;
    std::vector<owl::vec3f> normals;
    std::vector<float> areas;
    std::vector<float> area_cdf;

    // Loop over shapes
    if constexpr (Dim == 3) {
        vertices.resize(attrib.vertices.size() / 3);
        for (int v = 0; v < (int)attrib.vertices.size() / 3; v++)
            vertices[v] = config.obstacle_scale *
                          owl::vec3f(attrib.vertices[3 * v], attrib.vertices[3 * v + 1], attrib.vertices[3 * v + 2]);

        for (size_t s = 0; s < shapes.size(); s++) {
            indices.reserve(indices.size() + shapes[s].mesh.num_face_vertices.size());
            for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
                indices.emplace_back(
                    shapes[s].mesh.indices[3 * f].vertex_index, shapes[s].mesh.indices[3 * f + 1].vertex_index,
                    shapes[s].mesh.indices[3 * f + 2].vertex_index
                );
            }
        }

        num_primitives = indices.size();
        normals.resize(num_primitives);
        areas.resize(num_primitives);
        area_cdf.resize(num_primitives);

        double area_sum = 0.f;
        for (int f = 0; f < num_primitives; f++) {
            owl::vec3f v0 = vertices[indices[f].x];
            owl::vec3f v1 = vertices[indices[f].y];
            owl::vec3f v2 = vertices[indices[f].z];
            owl::vec3f e0 = v1 - v0;
            owl::vec3f e1 = v2 - v0;
            owl::vec3f n = cross(e1, e0);
            normals[f] = normalize(n);
            areas[f] = length(n) / 2.f;
            area_sum += areas[f];
            area_cdf[f] = area_sum;
        }
        for (int f = 0; f < num_primitives; f++) area_cdf[f] /= area_sum;
    } else if constexpr (Dim == 2) {
        for (size_t s = 0; s < shapes.size(); s++) {
            int index_offset = 0;
            for (size_t l = 0; l < shapes[s].lines.num_line_vertices.size(); l++) {
                for (size_t seg = 0; seg < (size_t)shapes[s].lines.num_line_vertices[l] - 1; seg++) {
                    owl::vec2f v0(
                        attrib.vertices[3 * shapes[s].lines.indices[index_offset + seg].vertex_index],
                        attrib.vertices[3 * shapes[s].lines.indices[index_offset + seg].vertex_index + 1]
                    );
                    owl::vec2f v1(
                        attrib.vertices[3 * shapes[s].lines.indices[index_offset + seg + 1].vertex_index],
                        attrib.vertices[3 * shapes[s].lines.indices[index_offset + seg + 1].vertex_index + 1]
                    );
                    v0 *= config.obstacle_scale;
                    v1 *= config.obstacle_scale;

                    vertices.emplace_back(v0.x, v0.y, 0.f);
                    vertices.emplace_back(v1.x, v1.y, -1.f);
                    vertices.emplace_back(v1.x, v1.y, 1.f);
                    indices.emplace_back(vertices.size() - 3, vertices.size() - 2, vertices.size() - 1);
                    owl::vec2f t = v1 - v0;
                    owl::vec2f n = owl::normalize(owl::vec2f(t.y, -t.x));
                    normals.emplace_back(owl::cast_dim<3>(n));
                    areas.emplace_back(owl::length(t));
                }
                index_offset += shapes[s].lines.num_line_vertices[l];
            }
        }

        num_primitives = indices.size();
        area_cdf.resize(num_primitives);
        double area_sum = 0.f;
        for (int f = 0; f < num_primitives; f++) {
            area_sum += areas[f];
            area_cdf[f] = area_sum;
        }
        for (int f = 0; f < num_primitives; f++) area_cdf[f] /= area_sum;
    }

    if (config.invert_normals)
        for (auto &n : normals) n = -n;

    std::cout << "#vertices: " << vertices.size() << std::endl;
    std::cout << "#primitives: " << indices.size() << std::endl;

    if (num_primitives == 0 && config.domain_type == DomainType::BoundedDomain) {
        std::cerr << "When no obstacles are specified, use UnboundedDomain domain type instead." << std::endl;
        exit(1);
    }

    /* Initialize pseudo boundary */
    int pseudo_boundary_num_primitives = 0;
    std::vector<owl::vec3f> pseudo_boundary_vertices;
    std::vector<owl::vec3i> pseudo_boundary_indices;
    std::vector<owl::vec3f> pseudo_boundary_normals;
    std::vector<float> pseudo_boundary_areas;
    std::vector<float> pseudo_boundary_area_cdf;
    if (config.domain_type == DomainType::UnboundedDomain) {
        if constexpr (Dim == 2) {
            owl::vec2f vertices[4] = {
                {config.domain_size[0] / 2, config.domain_size[1] / 2},
                {config.domain_size[0] / 2, -config.domain_size[1] / 2},
                {-config.domain_size[0] / 2, -config.domain_size[1] / 2},
                {-config.domain_size[0] / 2, config.domain_size[1] / 2}};
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
        } else if constexpr (Dim == 3) {
            pseudo_boundary_vertices.emplace_back(
                config.domain_size[0] / 2, config.domain_size[1] / 2, config.domain_size[2] / 2
            );
            pseudo_boundary_vertices.emplace_back(
                config.domain_size[0] / 2, config.domain_size[1] / 2, -config.domain_size[2] / 2
            );
            pseudo_boundary_vertices.emplace_back(
                config.domain_size[0] / 2, -config.domain_size[1] / 2, config.domain_size[2] / 2
            );
            pseudo_boundary_vertices.emplace_back(
                config.domain_size[0] / 2, -config.domain_size[1] / 2, -config.domain_size[2] / 2
            );
            pseudo_boundary_vertices.emplace_back(
                -config.domain_size[0] / 2, config.domain_size[1] / 2, config.domain_size[2] / 2
            );
            pseudo_boundary_vertices.emplace_back(
                -config.domain_size[0] / 2, config.domain_size[1] / 2, -config.domain_size[2] / 2
            );
            pseudo_boundary_vertices.emplace_back(
                -config.domain_size[0] / 2, -config.domain_size[1] / 2, config.domain_size[2] / 2
            );
            pseudo_boundary_vertices.emplace_back(
                -config.domain_size[0] / 2, -config.domain_size[1] / 2, -config.domain_size[2] / 2
            );

            pseudo_boundary_indices.emplace_back(0, 1, 2);
            pseudo_boundary_indices.emplace_back(3, 2, 1);
            pseudo_boundary_indices.emplace_back(4, 6, 5);
            pseudo_boundary_indices.emplace_back(7, 5, 6);
            pseudo_boundary_indices.emplace_back(0, 4, 1);
            pseudo_boundary_indices.emplace_back(5, 1, 4);
            pseudo_boundary_indices.emplace_back(2, 3, 6);
            pseudo_boundary_indices.emplace_back(7, 6, 3);
            pseudo_boundary_indices.emplace_back(0, 2, 4);
            pseudo_boundary_indices.emplace_back(6, 4, 2);
            pseudo_boundary_indices.emplace_back(1, 5, 3);
            pseudo_boundary_indices.emplace_back(7, 3, 5);

            for (int f = 0; f < (int)pseudo_boundary_indices.size(); f++) {
                const owl::vec3f &v0 = pseudo_boundary_vertices[pseudo_boundary_indices[f].x];
                const owl::vec3f &v1 = pseudo_boundary_vertices[pseudo_boundary_indices[f].y];
                const owl::vec3f &v2 = pseudo_boundary_vertices[pseudo_boundary_indices[f].z];
                const owl::vec3f e0 = v1 - v0;
                const owl::vec3f e1 = v2 - v0;
                const owl::vec3f n = cross(e1, e0);
                pseudo_boundary_normals.emplace_back(normalize(n));
                pseudo_boundary_areas.emplace_back(length(n) / 2.f);
            }
        }

        pseudo_boundary_num_primitives = pseudo_boundary_indices.size();
        pseudo_boundary_area_cdf.resize(pseudo_boundary_num_primitives);
        double area_sum = 0.f;
        for (int f = 0; f < pseudo_boundary_num_primitives; f++) {
            area_sum += pseudo_boundary_areas[f];
            pseudo_boundary_area_cdf[f] = area_sum;
        }
        for (int f = 0; f < pseudo_boundary_num_primitives; f++) pseudo_boundary_area_cdf[f] /= area_sum;
    }

    /* Initialize Oprix Data */
    std::cout << "Initializing OptiX..." << std::endl;

    context = owlContextCreate(nullptr, 0);
    if (context == NULL) {
        std::cerr << "Failed to create OptiX context." << std::endl;
        exit(1);
    }
    int num_gpus_found = owlGetDeviceCount(context);
    std::cout << num_gpus_found << " GPU(s) detected." << std::endl;
    owlContextSetRayTypeCount(context, (int)RayType::NumRayTypes);
    OWLModule geometry_module = owlModuleCreate(context, geometry_optix_ptx);
    OWLModule velocity_fluids_module = owlModuleCreate(context, velocity_fluids_optix_ptx);
    OWLModule fluids_common_module = owlModuleCreate(context, fluids_common_optix_ptx);

    OWLBuffer vertex_buffer = owlManagedMemoryBufferCreate(context, OWL_FLOAT3, vertices.size(), vertices.data());
    OWLBuffer index_buffer = owlManagedMemoryBufferCreate(context, OWL_INT3, indices.size(), indices.data());
    OWLBuffer normal_buffer = owlManagedMemoryBufferCreate(context, OWL_FLOAT3, normals.size(), normals.data());
    OWLBuffer area_buffer = owlManagedMemoryBufferCreate(context, OWL_FLOAT, areas.size(), areas.data());
    OWLBuffer area_cdf_buffer = owlManagedMemoryBufferCreate(context, OWL_FLOAT, area_cdf.size(), area_cdf.data());

    OWLBuffer pseudo_boundary_vertex_buffer = owlManagedMemoryBufferCreate(
        context, OWL_FLOAT3, pseudo_boundary_vertices.size(), pseudo_boundary_vertices.data()
    );
    OWLBuffer pseudo_boundary_index_buffer =
        owlManagedMemoryBufferCreate(context, OWL_INT3, pseudo_boundary_indices.size(), pseudo_boundary_indices.data());
    OWLBuffer pseudo_boundary_normal_buffer = owlManagedMemoryBufferCreate(
        context, OWL_FLOAT3, pseudo_boundary_normals.size(), pseudo_boundary_normals.data()
    );
    OWLBuffer pseudo_boundary_area_buffer =
        owlManagedMemoryBufferCreate(context, OWL_FLOAT, pseudo_boundary_areas.size(), pseudo_boundary_areas.data());
    OWLBuffer pseudo_boundary_area_cdf_buffer = owlManagedMemoryBufferCreate(
        context, OWL_FLOAT, pseudo_boundary_area_cdf.size(), pseudo_boundary_area_cdf.data()
    );

    world = {0};
    if (num_primitives) {
        // Only normal is used in AnyHit shaders.
        OWLVarDecl triangle_mesh_geo_vars[] = {{"normal", OWL_BUFPTR, 0}, {/* sentinel: */ nullptr}};
        OWLGeomType triangle_mesh_geo_type =
            owlGeomTypeCreate(context, OWL_TRIANGLES, sizeof(owl::vec3f *), triangle_mesh_geo_vars, -1);
        owlGeomTypeSetAnyHit(
            triangle_mesh_geo_type, (int)RayType::IntersectionSamplingRay, geometry_module,
            "RayIntersectionSamplingAnyHit"
        );
        owlGeomTypeSetAnyHit(
            triangle_mesh_geo_type, (int)RayType::WindingNumberRay, geometry_module, "WindingNumberAnyHit"
        );
        owlGeomTypeSetAnyHit(
            triangle_mesh_geo_type, (int)RayType::ClosestBoundaryPointRay, geometry_module, "ClosestBoundaryPointAnyHit"
        );
        OWLGeom triangle_mesh_geo = owlGeomCreate(context, triangle_mesh_geo_type);

        owlTrianglesSetVertices(triangle_mesh_geo, vertex_buffer, vertices.size(), sizeof(owl::vec3f), 0);
        owlTrianglesSetIndices(triangle_mesh_geo, index_buffer, indices.size(), sizeof(owl::vec3i), 0);
        owlGeomSetBuffer(triangle_mesh_geo, "normal", normal_buffer);

        OWLGroup triangle_mesh_group = owlTrianglesGeomGroupCreate(context, 1, &triangle_mesh_geo);
        owlGroupBuildAccel(triangle_mesh_group);
        world = owlInstanceGroupCreate(
            context, 1, &triangle_mesh_group, nullptr, nullptr, OWL_MATRIX_FORMAT_OWL, OPTIX_BUILD_FLAG_ALLOW_UPDATE
        );
        transform = owl::affine3f::translate(
            owl::vec3f(config.obstacle_shift[0], config.obstacle_shift[1], Dim == 2 ? 0.f : config.obstacle_shift[2])
        );
        owlInstanceGroupSetTransform(world, (int)0, (const float *)&transform, OWL_MATRIX_FORMAT_OWL);
        owlGroupBuildAccel(world);
    }

    OWLVarDecl projection_ray_gen_vars[] = {
        {"acc_structure", OWL_GROUP, OWL_OFFSETOF(ProjectionRayGenData<Dim>, acc_structure)},
        {"mesh.vertex_buf", OWL_BUFPTR, OWL_OFFSETOF(ProjectionRayGenData<Dim>, mesh.vertex_buf)},
        {"mesh.index_buf", OWL_BUFPTR, OWL_OFFSETOF(ProjectionRayGenData<Dim>, mesh.index_buf)},
        {"mesh.normal_buf", OWL_BUFPTR, OWL_OFFSETOF(ProjectionRayGenData<Dim>, mesh.normal_buf)},
        {"mesh.transform", OWL_USER_TYPE(owl::affine3f), OWL_OFFSETOF(ProjectionRayGenData<Dim>, mesh.transform)},
        {"mesh.area_buf", OWL_BUFPTR, OWL_OFFSETOF(ProjectionRayGenData<Dim>, mesh.area_buf)},
        {"mesh.cdf_buf", OWL_BUFPTR, OWL_OFFSETOF(ProjectionRayGenData<Dim>, mesh.cdf_buf)},
        {"mesh.num_primitives", OWL_INT, OWL_OFFSETOF(ProjectionRayGenData<Dim>, mesh.num_primitives)},
        {"pseudo_boundary_mesh.vertex_buf", OWL_BUFPTR,
         OWL_OFFSETOF(ProjectionRayGenData<Dim>, pseudo_boundary_mesh.vertex_buf)},
        {"pseudo_boundary_mesh.index_buf", OWL_BUFPTR,
         OWL_OFFSETOF(ProjectionRayGenData<Dim>, pseudo_boundary_mesh.index_buf)},
        {"pseudo_boundary_mesh.normal_buf", OWL_BUFPTR,
         OWL_OFFSETOF(ProjectionRayGenData<Dim>, pseudo_boundary_mesh.normal_buf)},
        {"pseudo_boundary_mesh.transform", OWL_USER_TYPE(owl::affine3f),
         OWL_OFFSETOF(ProjectionRayGenData<Dim>, pseudo_boundary_mesh.transform)},
        {"pseudo_boundary_mesh.area_buf", OWL_BUFPTR,
         OWL_OFFSETOF(ProjectionRayGenData<Dim>, pseudo_boundary_mesh.area_buf)},
        {"pseudo_boundary_mesh.cdf_buf", OWL_BUFPTR,
         OWL_OFFSETOF(ProjectionRayGenData<Dim>, pseudo_boundary_mesh.cdf_buf)},
        {"pseudo_boundary_mesh.num_primitives", OWL_INT,
         OWL_OFFSETOF(ProjectionRayGenData<Dim>, pseudo_boundary_mesh.num_primitives)},
        {"config", OWL_USER_TYPE(DeviceConfig<Dim>), OWL_OFFSETOF(ProjectionRayGenData<Dim>, config)},
        {"device_indx", OWL_DEVICE, OWL_OFFSETOF(ProjectionRayGenData<Dim>, device_index)},
        {"device_count", OWL_INT, OWL_OFFSETOF(ProjectionRayGenData<Dim>, device_count)},
        {"dt", OWL_FLOAT, OWL_OFFSETOF(ProjectionRayGenData<Dim>, dt)},

        {"evaluation_point_buf", OWL_RAW_POINTER, OWL_OFFSETOF(ProjectionRayGenData<Dim>, evaluation_point_buf)},
        {"winding_number_buf", OWL_RAW_POINTER, OWL_OFFSETOF(ProjectionRayGenData<Dim>, winding_number_buf)},
        {"new_velocity_buf", OWL_RAW_POINTER, OWL_OFFSETOF(ProjectionRayGenData<Dim>, new_velocity_buf)},
        {"random_state_buf", OWL_RAW_POINTER, OWL_OFFSETOF(ProjectionRayGenData<Dim>, random_state_buf)},
        {"num_evaluation_points", OWL_INT, OWL_OFFSETOF(ProjectionRayGenData<Dim>, num_evaluation_points)},

        {"grid_winding_number_buf", OWL_RAW_POINTER, OWL_OFFSETOF(ProjectionRayGenData<Dim>, grid_winding_number_buf)},
        {"grid_advection_velocity_buf", OWL_RAW_POINTER,
         OWL_OFFSETOF(ProjectionRayGenData<Dim>, grid_advection_velocity_buf)},
        {"grid_velocity_buf", OWL_RAW_POINTER, OWL_OFFSETOF(ProjectionRayGenData<Dim>, grid_velocity_buf)},
        {"solid_velocity", Dim == 3 ? OWL_FLOAT3 : OWL_FLOAT2, OWL_OFFSETOF(ProjectionRayGenData<Dim>, solid_velocity)},
        {"velocity_source_buf", OWL_RAW_POINTER, OWL_OFFSETOF(ProjectionRayGenData<Dim>, velocity_source_buf)},
        {"velocity_source_count", OWL_INT, OWL_OFFSETOF(ProjectionRayGenData<Dim>, velocity_source_count)},
        {"vpl_data_buf", OWL_RAW_POINTER, OWL_OFFSETOF(ProjectionRayGenData<Dim>, vpl_data_buf)},
        {/* sentinel: */ nullptr}};

    projection_ray_gen = owlRayGenCreate(
        context, velocity_fluids_module, Dim == 3 ? "velocityProject3DRayGen" : "velocityProject2DRayGen",
        sizeof(ProjectionRayGenData<Dim>), projection_ray_gen_vars, -1
    );

    projection_vpl_construct_ray_gen = owlRayGenCreate(
        context, velocity_fluids_module,
        Dim == 3 ? "velocityProjectVPLConstruct3DRayGen" : "velocityProjectVPLConstruct2DRayGen",
        sizeof(ProjectionRayGenData<Dim>), projection_ray_gen_vars, -1
    );

    projection_vpl_gather_ray_gen = owlRayGenCreate(
        context, velocity_fluids_module,
        Dim == 3 ? "velocityProjectVPLGather3DRayGen" : "velocityProjectVPLGather2DRayGen",
        sizeof(ProjectionRayGenData<Dim>), projection_ray_gen_vars, -1
    );

    owl::affine3f identity;
    for (auto &ray_gen : {projection_ray_gen, projection_vpl_construct_ray_gen, projection_vpl_gather_ray_gen}) {
        owlRayGenSetGroup(ray_gen, "acc_structure", world);
        owlRayGenSetBuffer(ray_gen, "mesh.vertex_buf", vertex_buffer);
        owlRayGenSetBuffer(ray_gen, "mesh.index_buf", index_buffer);
        owlRayGenSetBuffer(ray_gen, "mesh.normal_buf", normal_buffer);
        owlRayGenSetRaw(ray_gen, "mesh.transform", &transform);
        owlRayGenSetBuffer(ray_gen, "mesh.area_buf", area_buffer);
        owlRayGenSetBuffer(ray_gen, "mesh.cdf_buf", area_cdf_buffer);
        owlRayGenSet1i(ray_gen, "mesh.num_primitives", num_primitives);
        owlRayGenSetRaw(ray_gen, "config", &config);
        owlRayGenSet1i(ray_gen, "device_count", num_gpus_found);
        owlRayGenSet1i(ray_gen, "velocity_source_count", 0);
        owlRayGenSetBuffer(ray_gen, "pseudo_boundary_mesh.vertex_buf", pseudo_boundary_vertex_buffer);
        owlRayGenSetBuffer(ray_gen, "pseudo_boundary_mesh.index_buf", pseudo_boundary_index_buffer);
        owlRayGenSetBuffer(ray_gen, "pseudo_boundary_mesh.normal_buf", pseudo_boundary_normal_buffer);
        owlRayGenSetRaw(ray_gen, "pseudo_boundary_mesh.transform", &identity);
        owlRayGenSetBuffer(ray_gen, "pseudo_boundary_mesh.area_buf", pseudo_boundary_area_buffer);
        owlRayGenSetBuffer(ray_gen, "pseudo_boundary_mesh.cdf_buf", pseudo_boundary_area_cdf_buffer);
        owlRayGenSet1i(ray_gen, "pseudo_boundary_mesh.num_primitives", pseudo_boundary_num_primitives);
    }

    if (config.velocity_diffusion_coefficient > 0) {
#define COMMA ,
        OWLVarDecl vector_diffusion_ray_gen_vars[] = {
            {"acc_structure", OWL_GROUP,
             OWL_OFFSETOF(DiffusionRayGenData<Dim COMMA owl::vec_t<float COMMA Dim>>, acc_structure)},
            {"mesh.vertex_buf", OWL_BUFPTR,
             OWL_OFFSETOF(DiffusionRayGenData<Dim COMMA owl::vec_t<float COMMA Dim>>, mesh.vertex_buf)},
            {"mesh.index_buf", OWL_BUFPTR,
             OWL_OFFSETOF(DiffusionRayGenData<Dim COMMA owl::vec_t<float COMMA Dim>>, mesh.index_buf)},
            {"mesh.normal_buf", OWL_BUFPTR,
             OWL_OFFSETOF(DiffusionRayGenData<Dim COMMA owl::vec_t<float COMMA Dim>>, mesh.normal_buf)},
            {"mesh.transform", OWL_USER_TYPE(owl::affine3f),
             OWL_OFFSETOF(DiffusionRayGenData<Dim COMMA owl::vec_t<float COMMA Dim>>, mesh.transform)},
            {"mesh.area_buf", OWL_BUFPTR,
             OWL_OFFSETOF(DiffusionRayGenData<Dim COMMA owl::vec_t<float COMMA Dim>>, mesh.area_buf)},
            {"mesh.cdf_buf", OWL_BUFPTR,
             OWL_OFFSETOF(DiffusionRayGenData<Dim COMMA owl::vec_t<float COMMA Dim>>, mesh.cdf_buf)},
            {"mesh.num_primitives", OWL_INT,
             OWL_OFFSETOF(DiffusionRayGenData<Dim COMMA owl::vec_t<float COMMA Dim>>, mesh.num_primitives)},
            {"config", OWL_USER_TYPE(DeviceConfig<Dim>),
             OWL_OFFSETOF(DiffusionRayGenData<Dim COMMA owl::vec_t<float COMMA Dim>>, config)},
            {"device_indx", OWL_DEVICE,
             OWL_OFFSETOF(DiffusionRayGenData<Dim COMMA owl::vec_t<float COMMA Dim>>, device_index)},
            {"device_count", OWL_INT,
             OWL_OFFSETOF(DiffusionRayGenData<Dim COMMA owl::vec_t<float COMMA Dim>>, device_count)},
            {"dt", OWL_FLOAT, OWL_OFFSETOF(DiffusionRayGenData<Dim COMMA owl::vec_t<float COMMA Dim>>, dt)},
            {"diffusion_coefficient", OWL_FLOAT,
             OWL_OFFSETOF(DiffusionRayGenData<Dim COMMA owl::vec_t<float COMMA Dim>>, diffusion_coefficient)},

            {"use_initial_value_as_boundary_value", OWL_BOOL,
             OWL_OFFSETOF(
                 DiffusionRayGenData<Dim COMMA owl::vec_t<float COMMA Dim>>, use_initial_value_as_boundary_value
             )},
            {"boundary_value", Dim == 3 ? OWL_FLOAT3 : OWL_FLOAT2,
             OWL_OFFSETOF(DiffusionRayGenData<Dim COMMA owl::vec_t<float COMMA Dim>>, boundary_value)},

            {"evaluation_point_buf", OWL_RAW_POINTER,
             OWL_OFFSETOF(DiffusionRayGenData<Dim COMMA owl::vec_t<float COMMA Dim>>, evaluation_point_buf)},
            {"winding_number_buf", OWL_RAW_POINTER,
             OWL_OFFSETOF(DiffusionRayGenData<Dim COMMA owl::vec_t<float COMMA Dim>>, winding_number_buf)},
            {"new_field_buf", OWL_RAW_POINTER,
             OWL_OFFSETOF(DiffusionRayGenData<Dim COMMA owl::vec_t<float COMMA Dim>>, new_field_buf)},
            {"random_state_buf", OWL_RAW_POINTER,
             OWL_OFFSETOF(DiffusionRayGenData<Dim COMMA owl::vec_t<float COMMA Dim>>, random_state_buf)},
            {"num_evaluation_points", OWL_INT,
             OWL_OFFSETOF(DiffusionRayGenData<Dim COMMA owl::vec_t<float COMMA Dim>>, num_evaluation_points)},

            {"grid_winding_number_buf", OWL_RAW_POINTER,
             OWL_OFFSETOF(DiffusionRayGenData<Dim COMMA owl::vec_t<float COMMA Dim>>, grid_winding_number_buf)},
            {"grid_field_buf", OWL_RAW_POINTER,
             OWL_OFFSETOF(DiffusionRayGenData<Dim COMMA owl::vec_t<float COMMA Dim>>, grid_field_buf)},
            {"grid_advection_velocity_buf", OWL_RAW_POINTER,
             OWL_OFFSETOF(DiffusionRayGenData<Dim COMMA owl::vec_t<float COMMA Dim>>, grid_advection_velocity_buf)},
            {/* sentinel: */
             nullptr}};
        vector_diffusion_ray_gen = owlRayGenCreate(
            context, fluids_common_module, Dim == 3 ? "diffusionVector3DRayGen" : "diffusionVector2DRayGen",
            sizeof(DiffusionRayGenData<Dim, owl::vec_t<float, Dim>>), vector_diffusion_ray_gen_vars, -1
        );
#undef COMMA

        owlRayGenSetGroup(vector_diffusion_ray_gen, "acc_structure", world);
        owlRayGenSetBuffer(vector_diffusion_ray_gen, "mesh.vertex_buf", vertex_buffer);
        owlRayGenSetBuffer(vector_diffusion_ray_gen, "mesh.index_buf", index_buffer);
        owlRayGenSetBuffer(vector_diffusion_ray_gen, "mesh.normal_buf", normal_buffer);
        owlRayGenSetRaw(vector_diffusion_ray_gen, "mesh.transform", &transform);
        owlRayGenSetBuffer(vector_diffusion_ray_gen, "mesh.area_buf", area_buffer);
        owlRayGenSetBuffer(vector_diffusion_ray_gen, "mesh.cdf_buf", area_cdf_buffer);
        owlRayGenSet1i(vector_diffusion_ray_gen, "mesh.num_primitives", num_primitives);
        owlRayGenSetRaw(vector_diffusion_ray_gen, "config", &config);
        owlRayGenSet1i(vector_diffusion_ray_gen, "device_count", num_gpus_found);
    }

    if (config.concentration_diffusion_coefficient > 0 || config.temperature_diffusion_coefficient > 0) {
#define COMMA ,
        OWLVarDecl scalar_diffusion_ray_gen_vars[] = {
            {"acc_structure", OWL_GROUP, OWL_OFFSETOF(DiffusionRayGenData<Dim COMMA float>, acc_structure)},
            {"mesh.vertex_buf", OWL_BUFPTR, OWL_OFFSETOF(DiffusionRayGenData<Dim COMMA float>, mesh.vertex_buf)},
            {"mesh.index_buf", OWL_BUFPTR, OWL_OFFSETOF(DiffusionRayGenData<Dim COMMA float>, mesh.index_buf)},
            {"mesh.normal_buf", OWL_BUFPTR, OWL_OFFSETOF(DiffusionRayGenData<Dim COMMA float>, mesh.normal_buf)},
            {"mesh.transform", OWL_USER_TYPE(owl::affine3f),
             OWL_OFFSETOF(DiffusionRayGenData<Dim COMMA float>, mesh.transform)},
            {"mesh.area_buf", OWL_BUFPTR, OWL_OFFSETOF(DiffusionRayGenData<Dim COMMA float>, mesh.area_buf)},
            {"mesh.cdf_buf", OWL_BUFPTR, OWL_OFFSETOF(DiffusionRayGenData<Dim COMMA float>, mesh.cdf_buf)},
            {"mesh.num_primitives", OWL_INT, OWL_OFFSETOF(DiffusionRayGenData<Dim COMMA float>, mesh.num_primitives)},
            {"config", OWL_USER_TYPE(DeviceConfig<Dim>), OWL_OFFSETOF(DiffusionRayGenData<Dim COMMA float>, config)},
            {"device_indx", OWL_DEVICE, OWL_OFFSETOF(DiffusionRayGenData<Dim COMMA float>, device_index)},
            {"device_count", OWL_INT, OWL_OFFSETOF(DiffusionRayGenData<Dim COMMA float>, device_count)},
            {"dt", OWL_FLOAT, OWL_OFFSETOF(DiffusionRayGenData<Dim COMMA float>, dt)},
            {"diffusion_coefficient", OWL_FLOAT,
             OWL_OFFSETOF(DiffusionRayGenData<Dim COMMA float>, diffusion_coefficient)},

            {"use_initial_value_as_boundary_value", OWL_BOOL,
             OWL_OFFSETOF(DiffusionRayGenData<Dim COMMA float>, use_initial_value_as_boundary_value)},
            {"boundary_value", OWL_FLOAT, OWL_OFFSETOF(DiffusionRayGenData<Dim COMMA float>, boundary_value)},

            {"evaluation_point_buf", OWL_RAW_POINTER,
             OWL_OFFSETOF(DiffusionRayGenData<Dim COMMA float>, evaluation_point_buf)},
            {"winding_number_buf", OWL_RAW_POINTER,
             OWL_OFFSETOF(DiffusionRayGenData<Dim COMMA float>, winding_number_buf)},
            {"new_field_buf", OWL_RAW_POINTER, OWL_OFFSETOF(DiffusionRayGenData<Dim COMMA float>, new_field_buf)},
            {"random_state_buf", OWL_RAW_POINTER, OWL_OFFSETOF(DiffusionRayGenData<Dim COMMA float>, random_state_buf)},
            {"num_evaluation_points", OWL_INT,
             OWL_OFFSETOF(DiffusionRayGenData<Dim COMMA float>, num_evaluation_points)},

            {"grid_winding_number_buf", OWL_RAW_POINTER,
             OWL_OFFSETOF(DiffusionRayGenData<Dim COMMA float>, grid_winding_number_buf)},
            {"grid_field_buf", OWL_RAW_POINTER, OWL_OFFSETOF(DiffusionRayGenData<Dim COMMA float>, grid_field_buf)},
            {"grid_advection_velocity_buf", OWL_RAW_POINTER,
             OWL_OFFSETOF(DiffusionRayGenData<Dim COMMA float>, grid_advection_velocity_buf)},
            {/* sentinel: */
             nullptr}};
        scalar_diffusion_ray_gen = owlRayGenCreate(
            context, fluids_common_module, Dim == 3 ? "diffusionScalar3DRayGen" : "diffusionScalar2DRayGen",
            sizeof(DiffusionRayGenData<Dim, owl::vec_t<float, Dim>>), scalar_diffusion_ray_gen_vars, -1
        );

        owlRayGenSetGroup(scalar_diffusion_ray_gen, "acc_structure", world);
        owlRayGenSetBuffer(scalar_diffusion_ray_gen, "mesh.vertex_buf", vertex_buffer);
        owlRayGenSetBuffer(scalar_diffusion_ray_gen, "mesh.index_buf", index_buffer);
        owlRayGenSetBuffer(scalar_diffusion_ray_gen, "mesh.normal_buf", normal_buffer);
        owlRayGenSetRaw(scalar_diffusion_ray_gen, "mesh.transform", &transform);
        owlRayGenSetBuffer(scalar_diffusion_ray_gen, "mesh.area_buf", area_buffer);
        owlRayGenSetBuffer(scalar_diffusion_ray_gen, "mesh.cdf_buf", area_cdf_buffer);
        owlRayGenSet1i(scalar_diffusion_ray_gen, "mesh.num_primitives", num_primitives);
        owlRayGenSetRaw(scalar_diffusion_ray_gen, "config", &config);
        owlRayGenSet1i(scalar_diffusion_ray_gen, "device_count", num_gpus_found);
#undef COMMA
    }

    OWLVarDecl winding_number_ray_gen_vars[] = {
        {"acc_structure", OWL_GROUP, OWL_OFFSETOF(WindingNumberRayGenData<Dim>, acc_structure)},
        {"config", OWL_USER_TYPE(DeviceConfig<Dim>), OWL_OFFSETOF(WindingNumberRayGenData<Dim>, config)},
        {"num_primitives", OWL_INT, OWL_OFFSETOF(WindingNumberRayGenData<Dim>, num_primitives)},
        {"device_indx", OWL_DEVICE, OWL_OFFSETOF(WindingNumberRayGenData<Dim>, device_index)},
        {"device_count", OWL_INT, OWL_OFFSETOF(WindingNumberRayGenData<Dim>, device_count)},
        {"evaluation_point_buf", OWL_RAW_POINTER, OWL_OFFSETOF(WindingNumberRayGenData<Dim>, evaluation_point_buf)},
        {"winding_number_buf", OWL_RAW_POINTER, OWL_OFFSETOF(WindingNumberRayGenData<Dim>, winding_number_buf)},
        {"random_state_buf", OWL_RAW_POINTER, OWL_OFFSETOF(WindingNumberRayGenData<Dim>, random_state_buf)},
        {"num_evaluation_points", OWL_INT, OWL_OFFSETOF(WindingNumberRayGenData<Dim>, num_evaluation_points)},
        {/* sentinel: */ nullptr}};
    winding_number_ray_gen = owlRayGenCreate(
        context, geometry_module, Dim == 3 ? "windingNumber3DRayGen" : "windingNumber2DRayGen",
        sizeof(WindingNumberRayGenData<Dim>), winding_number_ray_gen_vars, -1
    );

    owlRayGenSetGroup(winding_number_ray_gen, "acc_structure", world);
    owlRayGenSet1i(winding_number_ray_gen, "num_primitives", num_primitives);
    owlRayGenSetRaw(winding_number_ray_gen, "config", &config);
    owlRayGenSet1i(winding_number_ray_gen, "device_count", num_gpus_found);

    owlBuildPrograms(context);
    owlBuildPipeline(context);
    owlBuildSBT(context);
}

template <int Dim>
void Scene<Dim>::initialize_evaluation_points(thrust::universal_vector<owl::vec_t<float, Dim>> &evaluation_point) {
    evaluation_point.resize(config.num_evaluation_points);

    owl::vec_t<float, Dim> *evaluation_point_ptr = evaluation_point.data().get();
    const owl::vec_t<int, Dim> grid_res = config.grid_res;
    const owl::vec_t<float, Dim> domain_size = config.domain_size;
    thrust::for_each(
        thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(config.num_evaluation_points),
        [=] __device__(int idx) {
            evaluation_point_ptr[idx] = utils::idx_to_domain_point<Dim>(idx, grid_res, domain_size);
        }
    );
}

template <int Dim>
void Scene<Dim>::winding_number(
    const thrust::universal_vector<owl::vec_t<float, Dim>> &evaluation_point,
    thrust::universal_vector<float> &winding_number_buf, thrust::universal_vector<utils::randState_t> &random_state
) {
    VELMCFLUIDS_TIME_BEGIN

    winding_number_buf.resize(evaluation_point.size());
    owlRayGenSetPointer(winding_number_ray_gen, "evaluation_point_buf", evaluation_point.data().get());
    owlRayGenSetPointer(winding_number_ray_gen, "winding_number_buf", winding_number_buf.data().get());
    owlRayGenSetPointer(winding_number_ray_gen, "random_state_buf", random_state.data().get());
    owlRayGenSet1i(winding_number_ray_gen, "num_evaluation_points", evaluation_point.size());

    OWLParams lp = owlParamsCreate(context, 0, nullptr, -1);
    owlBuildSBT(context);

    // Launching more than necessary is fine as the kernel will terminate early for extra threads.
    owlLaunch3D(winding_number_ray_gen, 1024, 1024, (evaluation_point.size() + 1024 * 1024) / (1024 * 1024), lp);

    VELMCFLUIDS_TIME_END
}

template <int Dim>
void Scene<Dim>::project(
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

    const thrust::universal_vector<owl::vec_t<float, Dim + 1>> &velocity_sources
) {
    VELMCFLUIDS_TIME_BEGIN

    if (config.enable_vpl)
        project_vpl(
            evaluation_point, winding_number_buf, new_velocity, random_state, grid_winding_number_buf, grid_velocity,
            grid_advection_velocity, vpl_data, vpl_random_state, dt, enable_inplace_advection, velocity_sources
        );
    else
        project_pointwise(
            evaluation_point, winding_number_buf, new_velocity, random_state, grid_winding_number_buf, grid_velocity,
            grid_advection_velocity, dt, enable_inplace_advection, velocity_sources
        );

    VELMCFLUIDS_TIME_END
}

template <int Dim>
void Scene<Dim>::project_pointwise(
    const thrust::universal_vector<owl::vec_t<float, Dim>> &evaluation_point,
    const thrust::universal_vector<float> &winding_number_buf,
    thrust::universal_vector<owl::vec_t<float, Dim>> &new_velocity,
    thrust::universal_vector<utils::randState_t> &random_state,

    const thrust::universal_vector<float> &grid_winding_number_buf,
    const thrust::universal_vector<owl::vec_t<float, Dim>> &grid_velocity,
    const thrust::universal_vector<owl::vec_t<float, Dim>> &grid_advection_velocity,

    const float dt, const bool enable_inplace_advection,

    const thrust::universal_vector<owl::vec_t<float, Dim + 1>> &velocity_sources
) {
    VELMCFLUIDS_TIME_BEGIN

    if (new_velocity.data().get() == grid_velocity.data().get()) {
        std::cerr << "In-placeprojection is not supported." << std::endl;
        exit(1);
    }

    new_velocity.resize(evaluation_point.size());
    if (random_state.size() != evaluation_point.size()) {
        random_state.resize(evaluation_point.size());
        utils::random_states_init(random_state);
    }

    owlRayGenSet1f(projection_ray_gen, "dt", dt);
    owlRayGenSetRaw(projection_ray_gen, "mesh.transform", &transform);
    owlRayGenSetPointer(projection_ray_gen, "evaluation_point_buf", evaluation_point.data().get());
    owlRayGenSetPointer(projection_ray_gen, "winding_number_buf", winding_number_buf.data().get());
    owlRayGenSetPointer(projection_ray_gen, "new_velocity_buf", new_velocity.data().get());
    owlRayGenSetPointer(projection_ray_gen, "random_state_buf", random_state.data().get());
    owlRayGenSet1i(projection_ray_gen, "num_evaluation_points", evaluation_point.size());

    owlRayGenSetPointer(projection_ray_gen, "grid_winding_number_buf", grid_winding_number_buf.data().get());
    owlRayGenSetPointer(projection_ray_gen, "grid_velocity_buf", grid_velocity.data().get());
    owlRayGenSetPointer(
        projection_ray_gen, "grid_advection_velocity_buf",
        enable_inplace_advection ? grid_advection_velocity.data().get() : nullptr
    );

    if constexpr (Dim == 3) {
        owlRayGenSet3f(
            projection_ray_gen, "solid_velocity", config.solid_velocity[0], config.solid_velocity[1],
            config.solid_velocity[2]
        );
    } else if constexpr (Dim == 2) {
        owlRayGenSet2f(projection_ray_gen, "solid_velocity", config.solid_velocity[0], config.solid_velocity[1]);
    }
    owlRayGenSetPointer(projection_ray_gen, "velocity_source_buf", velocity_sources.data().get());
    owlRayGenSet1i(projection_ray_gen, "velocity_source_count", velocity_sources.size());

    OWLParams lp = owlParamsCreate(context, 0, nullptr, -1);
    owlBuildSBT(context);

    // Launching more than necessary is fine as the kernel will terminate early for extra threads.
    owlLaunch3D(projection_ray_gen, 1024, 1024, (evaluation_point.size() + 1024 * 1024) / (1024 * 1024), lp);

    VELMCFLUIDS_TIME_END
};

template <int Dim>
void Scene<Dim>::project_vpl_construct(
    thrust::universal_vector<ProjectionVPLRecord<Dim>> &vpl_data,
    thrust::universal_vector<utils::randState_t> &vpl_random_state,

    const thrust::universal_vector<float> &grid_winding_number_buf,
    const thrust::universal_vector<owl::vec_t<float, Dim>> &grid_velocity,
    const thrust::universal_vector<owl::vec_t<float, Dim>> &grid_advection_velocity,

    const float dt, const bool enable_inplace_advection,

    const thrust::universal_vector<owl::vec_t<float, Dim + 1>> &velocity_sources
) {
    VELMCFLUIDS_TIME_BEGIN

    vpl_data.resize(config.num_path_samples * (config.path_length + 1));
    if (vpl_random_state.size() != (unsigned int)config.num_path_samples) {
        vpl_random_state.resize(config.num_path_samples);
        utils::random_states_init(vpl_random_state);
    }

    owlRayGenSet1f(projection_vpl_construct_ray_gen, "dt", dt);
    owlRayGenSetRaw(projection_vpl_construct_ray_gen, "mesh.transform", &transform);
    owlRayGenSetPointer(projection_vpl_construct_ray_gen, "random_state_buf", vpl_random_state.data().get());

    owlRayGenSetPointer(
        projection_vpl_construct_ray_gen, "grid_winding_number_buf", grid_winding_number_buf.data().get()
    );
    owlRayGenSetPointer(projection_vpl_construct_ray_gen, "grid_velocity_buf", grid_velocity.data().get());
    owlRayGenSetPointer(
        projection_vpl_construct_ray_gen, "grid_advection_velocity_buf",
        enable_inplace_advection ? grid_advection_velocity.data().get() : nullptr
    );

    if constexpr (Dim == 3) {
        owlRayGenSet3f(
            projection_vpl_construct_ray_gen, "solid_velocity", config.solid_velocity[0], config.solid_velocity[1],
            config.solid_velocity[2]
        );
    } else if constexpr (Dim == 2) {
        owlRayGenSet2f(
            projection_vpl_construct_ray_gen, "solid_velocity", config.solid_velocity[0], config.solid_velocity[1]
        );
    }
    owlRayGenSetPointer(projection_vpl_construct_ray_gen, "velocity_source_buf", velocity_sources.data().get());
    owlRayGenSet1i(projection_vpl_construct_ray_gen, "velocity_source_count", velocity_sources.size());
    owlRayGenSetPointer(projection_vpl_construct_ray_gen, "vpl_data_buf", vpl_data.data().get());

    OWLParams lp = owlParamsCreate(context, 0, nullptr, -1);
    owlBuildSBT(context);

    // Launching more than necessary is fine as the kernel will terminate early for extra threads.
    owlLaunch3D(
        projection_vpl_construct_ray_gen, 1024, 1024, (config.num_path_samples + 1024 * 1024) / (1024 * 1024), lp
    );

    VELMCFLUIDS_TIME_END
}

template <int Dim>
void Scene<Dim>::project_vpl_gather(
    const thrust::universal_vector<owl::vec_t<float, Dim>> &evaluation_point,
    const thrust::universal_vector<float> &winding_number_buf,
    thrust::universal_vector<owl::vec_t<float, Dim>> &new_velocity,
    thrust::universal_vector<utils::randState_t> &random_state,

    const thrust::universal_vector<float> &grid_winding_number_buf,
    const thrust::universal_vector<owl::vec_t<float, Dim>> &grid_velocity,
    const thrust::universal_vector<owl::vec_t<float, Dim>> &grid_advection_velocity,
    const thrust::universal_vector<ProjectionVPLRecord<Dim>> &vpl_data,

    const float dt, const bool enable_inplace_advection,

    const thrust::universal_vector<owl::vec_t<float, Dim + 1>> &velocity_sources
) {
    VELMCFLUIDS_TIME_BEGIN

    if (new_velocity.data().get() == grid_velocity.data().get()) {
        std::cerr << "In-placeprojection is not supported." << std::endl;
        exit(1);
    }

    new_velocity.resize(evaluation_point.size());
    if (random_state.size() != evaluation_point.size()) {
        random_state.resize(evaluation_point.size());
        utils::random_states_init(random_state);
    }

    owlRayGenSet1f(projection_vpl_gather_ray_gen, "dt", dt);
    owlRayGenSetRaw(projection_vpl_gather_ray_gen, "mesh.transform", &transform);
    owlRayGenSetPointer(projection_vpl_gather_ray_gen, "evaluation_point_buf", evaluation_point.data().get());
    owlRayGenSetPointer(projection_vpl_gather_ray_gen, "winding_number_buf", winding_number_buf.data().get());
    owlRayGenSetPointer(projection_vpl_gather_ray_gen, "new_velocity_buf", new_velocity.data().get());
    owlRayGenSetPointer(projection_vpl_gather_ray_gen, "random_state_buf", random_state.data().get());
    owlRayGenSet1i(projection_vpl_gather_ray_gen, "num_evaluation_points", evaluation_point.size());

    owlRayGenSetPointer(projection_vpl_gather_ray_gen, "grid_winding_number_buf", grid_winding_number_buf.data().get());
    owlRayGenSetPointer(projection_vpl_gather_ray_gen, "grid_velocity_buf", grid_velocity.data().get());
    owlRayGenSetPointer(
        projection_vpl_gather_ray_gen, "grid_advection_velocity_buf",
        enable_inplace_advection ? grid_advection_velocity.data().get() : nullptr
    );

    if constexpr (Dim == 3) {
        owlRayGenSet3f(
            projection_vpl_gather_ray_gen, "solid_velocity", config.solid_velocity[0], config.solid_velocity[1],
            config.solid_velocity[2]
        );
    } else if constexpr (Dim == 2) {
        owlRayGenSet2f(
            projection_vpl_gather_ray_gen, "solid_velocity", config.solid_velocity[0], config.solid_velocity[1]
        );
    }
    owlRayGenSetPointer(projection_vpl_gather_ray_gen, "velocity_source_buf", velocity_sources.data().get());
    owlRayGenSet1i(projection_vpl_gather_ray_gen, "velocity_source_count", velocity_sources.size());
    owlRayGenSetPointer(projection_vpl_gather_ray_gen, "vpl_data_buf", vpl_data.data().get());

    OWLParams lp = owlParamsCreate(context, 0, nullptr, -1);
    owlBuildSBT(context);

    // Launching more than necessary is fine as the kernel will terminate early for extra threads.
    owlLaunch3D(projection_vpl_gather_ray_gen, 1024, 1024, (evaluation_point.size() + 1024 * 1024) / (1024 * 1024), lp);

    VELMCFLUIDS_TIME_END
}

template <int Dim>
void Scene<Dim>::project_vpl(
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

    const thrust::universal_vector<owl::vec_t<float, Dim + 1>> &velocity_sources
) {
    VELMCFLUIDS_TIME_BEGIN

    project_vpl_construct(
        vpl_data, vpl_random_state, grid_winding_number_buf, grid_velocity, grid_advection_velocity, dt,
        enable_inplace_advection, velocity_sources
    );
    project_vpl_gather(
        evaluation_point, winding_number_buf, new_velocity, random_state, grid_winding_number_buf, grid_velocity,
        grid_advection_velocity, vpl_data, dt, enable_inplace_advection, velocity_sources
    );

    VELMCFLUIDS_TIME_END
}

template <int Dim>
void Scene<Dim>::diffuse_velocity(
    const thrust::universal_vector<owl::vec_t<float, Dim>> &evaluation_point,
    const thrust::universal_vector<float> &winding_number_buf,
    thrust::universal_vector<owl::vec_t<float, Dim>> &new_velocity,
    thrust::universal_vector<utils::randState_t> &random_state,

    const thrust::universal_vector<float> &grid_winding_number_buf,
    const thrust::universal_vector<owl::vec_t<float, Dim>> &grid_velocity,
    const thrust::universal_vector<owl::vec_t<float, Dim>> &grid_advection_velocity,

    float diffusion_coefficient, const float dt, const bool enable_inplace_advection
) {
    VELMCFLUIDS_TIME_BEGIN

    if (new_velocity.data().get() == grid_velocity.data().get()) {
        std::cerr << "In-place velocity diffusion is not supported." << std::endl;
        exit(1);
    }

    new_velocity.resize(evaluation_point.size());
    if (random_state.size() < evaluation_point.size()) {
        random_state.resize(evaluation_point.size());
        utils::random_states_init(random_state);
    }

    owlRayGenSetRaw(vector_diffusion_ray_gen, "mesh.transform", &transform);
    owlRayGenSet1f(vector_diffusion_ray_gen, "dt", dt);
    owlRayGenSet1f(vector_diffusion_ray_gen, "diffusion_coefficient", diffusion_coefficient);

    owlRayGenSet1b(vector_diffusion_ray_gen, "use_initial_value_as_boundary_value", false);
    if constexpr (Dim == 3) {
        owlRayGenSet3f(
            vector_diffusion_ray_gen, "boundary_value", config.solid_velocity[0], config.solid_velocity[1],
            config.solid_velocity[2]
        );
    } else if constexpr (Dim == 2) {
        owlRayGenSet2f(vector_diffusion_ray_gen, "boundary_value", config.solid_velocity[0], config.solid_velocity[1]);
    }

    owlRayGenSetPointer(vector_diffusion_ray_gen, "evaluation_point_buf", evaluation_point.data().get());
    owlRayGenSetPointer(vector_diffusion_ray_gen, "winding_number_buf", winding_number_buf.data().get());
    owlRayGenSetPointer(vector_diffusion_ray_gen, "new_field_buf", new_velocity.data().get());
    owlRayGenSetPointer(vector_diffusion_ray_gen, "random_state_buf", random_state.data().get());
    owlRayGenSet1i(vector_diffusion_ray_gen, "num_evaluation_points", evaluation_point.size());

    owlRayGenSetPointer(vector_diffusion_ray_gen, "grid_winding_number_buf", grid_winding_number_buf.data().get());
    owlRayGenSetPointer(vector_diffusion_ray_gen, "grid_field_buf", grid_velocity.data().get());
    owlRayGenSetPointer(
        vector_diffusion_ray_gen, "grid_advection_velocity_buf",
        enable_inplace_advection ? grid_advection_velocity.data().get() : nullptr
    );

    OWLParams lp = owlParamsCreate(context, 0, nullptr, -1);
    owlBuildSBT(context);

    // Launching more than necessary is fine as the kernel will terminate early for extra threads.
    owlLaunch3D(vector_diffusion_ray_gen, 1024, 1024, (evaluation_point.size() + 1024 * 1024) / (1024 * 1024), lp);

    VELMCFLUIDS_TIME_END
}

template <int Dim>
void Scene<Dim>::diffuse_scalar(
    const thrust::universal_vector<owl::vec_t<float, Dim>> &evaluation_point,
    const thrust::universal_vector<float> &winding_number_buf, thrust::universal_vector<float> &new_field,
    thrust::universal_vector<utils::randState_t> &random_state,

    const thrust::universal_vector<float> &grid_winding_number_buf, const thrust::universal_vector<float> &grid_field,
    const thrust::universal_vector<owl::vec_t<float, Dim>> &grid_advection_velocity,

    float diffusion_coefficient, const float dt, const bool enable_inplace_advection
) {
    VELMCFLUIDS_TIME_BEGIN

    if (new_field.data().get() == grid_field.data().get()) {
        std::cerr << "In-place scalar diffusion is not supported." << std::endl;
        exit(1);
    }

    new_field.resize(evaluation_point.size());
    if (random_state.size() < evaluation_point.size()) {
        random_state.resize(evaluation_point.size());
        utils::random_states_init(random_state);
    }

    owlRayGenSetRaw(scalar_diffusion_ray_gen, "mesh.transform", &transform);
    owlRayGenSet1f(scalar_diffusion_ray_gen, "dt", dt);
    owlRayGenSet1f(scalar_diffusion_ray_gen, "diffusion_coefficient", diffusion_coefficient);

    owlRayGenSet1b(scalar_diffusion_ray_gen, "use_initial_value_as_boundary_value", true);

    owlRayGenSetPointer(scalar_diffusion_ray_gen, "evaluation_point_buf", evaluation_point.data().get());
    owlRayGenSetPointer(scalar_diffusion_ray_gen, "winding_number_buf", winding_number_buf.data().get());
    owlRayGenSetPointer(scalar_diffusion_ray_gen, "new_field_buf", new_field.data().get());
    owlRayGenSetPointer(scalar_diffusion_ray_gen, "random_state_buf", random_state.data().get());
    owlRayGenSet1i(scalar_diffusion_ray_gen, "num_evaluation_points", evaluation_point.size());

    owlRayGenSetPointer(scalar_diffusion_ray_gen, "grid_winding_number_buf", grid_winding_number_buf.data().get());
    owlRayGenSetPointer(scalar_diffusion_ray_gen, "grid_field_buf", grid_field.data().get());
    owlRayGenSetPointer(
        scalar_diffusion_ray_gen, "grid_advection_velocity_buf",
        enable_inplace_advection ? grid_advection_velocity.data().get() : nullptr
    );

    OWLParams lp = owlParamsCreate(context, 0, nullptr, -1);
    owlBuildSBT(context);

    // Launching more than necessary is fine as the kernel will terminate early for extra threads.
    owlLaunch3D(scalar_diffusion_ray_gen, 1024, 1024, (evaluation_point.size() + 1024 * 1024) / (1024 * 1024), lp);

    VELMCFLUIDS_TIME_END
}

template <int Dim>
template <typename FieldValueType>
void Scene<Dim>::advect(
    const thrust::universal_vector<owl::vec_t<float, Dim>> &evaluation_point,
    const thrust::universal_vector<owl::vec_t<float, Dim>> &velocity,
    const thrust::universal_vector<FieldValueType> &field, thrust::universal_vector<FieldValueType> &new_field,
    const float dt
) {
    VELMCFLUIDS_TIME_BEGIN

    new_field.resize(field.size());

    const owl::vec_t<float, Dim> *evaluation_point_ptr = evaluation_point.data().get();
    const owl::vec_t<float, Dim> *velocity_ptr = velocity.data().get();
    const FieldValueType *field_ptr = field.data().get();
    FieldValueType *new_field_ptr = new_field.data().get();

    if (new_field_ptr == field_ptr) {
        std::cerr << "In-place advection is not supported." << std::endl;
        exit(1);
    }

    const DeviceConfig<Dim> device_config = config;

    thrust::for_each(
        thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(field.size()),
        [=] __device__(int idx) {
            const owl::vec_t<float, Dim> pos = evaluation_point_ptr[idx];
            new_field_ptr[idx] = get_advected_value(pos, velocity_ptr, field_ptr, dt, device_config);
        }
    );

    VELMCFLUIDS_TIME_END
}

template <int Dim> void __device__ atomicAdd(owl::vec_t<float, Dim> *address, const owl::vec_t<float, Dim> &val) {
    for (int i = 0; i < Dim; i++) atomicAdd(&((*address)[i]), val[i]);
}

template <int Dim>
template <typename FieldValueType>
void Scene<Dim>::particle_to_grid(
    const thrust::universal_vector<owl::vec_t<float, Dim>> &particle_positions,
    const thrust::universal_vector<FieldValueType> &particle_values,
    thrust::universal_vector<FieldValueType> &grid_field, thrust::universal_vector<float> &grid_weight_sum, int order
) {
    assert(order == 1 || order == 2 || order == 3);

    if (grid_weight_sum.size() != grid_field.size()) grid_weight_sum.resize(grid_field.size());

    thrust::fill(grid_field.begin(), grid_field.end(), utils::zero<FieldValueType>());
    thrust::fill(grid_weight_sum.begin(), grid_weight_sum.end(), 0);

    const owl::vec_t<float, Dim> *particle_positions_ptr = particle_positions.data().get();
    const FieldValueType *particle_values_ptr = particle_values.data().get();

    FieldValueType *grid_field_ptr = grid_field.data().get();
    float *grid_weight_sum_ptr = grid_weight_sum.data().get();

    const owl::vec_t<int, Dim> grid_res = config.grid_res;
    const owl::vec_t<float, Dim> domain_size = config.domain_size;

    // accumulate particle values to grid nodes
    thrust::for_each(
        thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(particle_positions.size()),
        [=] __device__(int particle_idx) {
            // for (int particle_idx = 0; particle_idx < particle_positions.size(); particle_idx++) {
            const owl::vec_t<float, Dim> pos = particle_positions_ptr[particle_idx];
            const FieldValueType field_value = particle_values_ptr[particle_idx];

            const owl::vec_t<float, Dim> idx = utils::domain_point_to_idx(pos, grid_res, domain_size);

            const auto h_func = (order == 1) ?
                    [](const float r) {
                        const float r_abs = abs(r);
                        if (-1.f < r && r < 1.f)
                            return 1.f - r_abs;
                        else
                            return 0.f;
                    }:
                (order == 2) ?
                    [](const float r) {
                        const float r_abs = abs(r);
                        if (-0.5f < r && r < 0.5f)
                            return 0.75f - r * r;
                        else if (-1.5f < r && r < 1.5f)
                            return 0.5f * (r_abs - 1.5f) * (r_abs - 1.5f);
                        else
                            return 0.f;
                    }:
                [](const float r) {
                    const float r_abs = abs(r);
                    if (-1 < r && r < 1)
                        return 0.5f * r_abs * r_abs * r_abs - r * r + 2.f / 3.f;
                    else if (-2 < r && r < 2)
                        return -1.f / 6.f * r_abs * r_abs * r_abs + r * r - 2.f * r_abs + (4.f / 3.f);
                    else
                        return 0.f;
                };

            const float max_radius = (order == 1) ? 1.f : (order == 2) ? 1.5f : 2.f;

            const auto accumulate_grid_node = [=](int grid_node_idx, float k) {
                atomicAdd(&grid_field_ptr[grid_node_idx], k * field_value);
                atomicAdd(&grid_weight_sum_ptr[grid_node_idx], k);
            };

            if constexpr (Dim == 2) {
                for (int x_idx = floorf(idx[0] - max_radius); x_idx <= ceilf(idx[0] + max_radius); x_idx++) {
                    if (x_idx < 0 || x_idx >= grid_res[0]) continue;
                    for (int y_idx = floorf(idx[1] - max_radius); y_idx <= ceilf(idx[1] + max_radius); y_idx++) {
                        if (y_idx < 0 || y_idx >= grid_res[1]) continue;

                        const owl::vec_t<float, Dim> idx_diff = idx - owl::vec_t<float, Dim>(x_idx, y_idx);
                        const float k = h_func(idx_diff[0]) * h_func(idx_diff[1]);

                        const int grid_node_idx = utils::flatten(owl::vec2i(x_idx, y_idx), grid_res);
                        accumulate_grid_node(grid_node_idx, k);
                    }
                }
            } else if constexpr (Dim == 3) {
                for (int x_idx = floorf(idx[0] - max_radius); x_idx <= ceilf(idx[0] + max_radius); x_idx++) {
                    if (x_idx < 0 || x_idx >= grid_res[0]) continue;
                    for (int y_idx = floorf(idx[1] - max_radius); y_idx <= ceilf(idx[1] + max_radius); y_idx++) {
                        if (y_idx < 0 || y_idx >= grid_res[1]) continue;
                        for (int z_idx = floorf(idx[2] - max_radius); z_idx <= ceilf(idx[2] + max_radius); z_idx++) {
                            if (z_idx < 0 || z_idx >= grid_res[2]) continue;

                            const owl::vec_t<float, Dim> idx_diff = idx - owl::vec_t<float, Dim>(x_idx, y_idx, z_idx);
                            const float k = h_func(idx_diff[0]) * h_func(idx_diff[1]) * h_func(idx_diff[2]);

                            const int grid_node_idx = utils::flatten(owl::vec3i(x_idx, y_idx, z_idx), grid_res);
                            accumulate_grid_node(grid_node_idx, k);
                        }
                    }
                }
            }
        }
    );

    // normalize the accumulated grid nodes
    thrust::for_each(
        thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(grid_field.size()),
        [=] __device__(int grid_node_idx) {
            if (grid_weight_sum_ptr[grid_node_idx] > 0.0f)
                grid_field_ptr[grid_node_idx] /= grid_weight_sum_ptr[grid_node_idx];
        }
    );
}

template <int Dim> void Scene<Dim>::setTranslation(const owl::vec_t<float, Dim> &translation) {
    this->transform = owl::affine3f::translate(owl::cast_dim<3>(translation));
    if (world == nullptr) return;
    // assuming there is one child instance group
    owlInstanceGroupSetTransform(world, (int)0, (const float *)&transform, OWL_MATRIX_FORMAT_OWL);
    owlGroupRefitAccel(world);
}

template <int Dim>
template <typename FieldValueType>
void Scene<Dim>::grid_breadth_first_fill(
    const thrust::universal_vector<float> &grid_winding_number_buf, thrust::universal_vector<FieldValueType> &grid_field
) {
    const float winding_number_threshold = 0.5f;
    thrust::device_vector<float> indicator_src_field = grid_winding_number_buf;
    thrust::device_vector<float> indicator_dst_field = grid_winding_number_buf;
    thrust::device_vector<FieldValueType> src_field = grid_field;
    thrust::device_vector<FieldValueType> dst_field = grid_field;

    // indicator value 0 corresponds to visited nodes
    {
        using namespace thrust::placeholders;
        if (config.domain_type == DomainType::UnboundedDomain) {
            // fluid domain 0 - >0
            // solid domain -1 -> 1
            thrust::for_each(indicator_src_field.begin(), indicator_src_field.end(), _1 = -_1);
            thrust::for_each(indicator_dst_field.begin(), indicator_dst_field.end(), _1 = -_1);
        } else {
            // fluid domain 1 - >0
            // solid domain 0 -> 1
            thrust::for_each(indicator_src_field.begin(), indicator_src_field.end(), _1 = -_1 + 1.0f);
            thrust::for_each(indicator_dst_field.begin(), indicator_dst_field.end(), _1 = -_1 + 1.0f);
        }
    }

    owl::vec_t<int, Dim> grid_res = config.grid_res;

    while (true) {
        float *indicator_src_ptr = indicator_src_field.data().get();
        float *indicator_dst_ptr = indicator_dst_field.data().get();
        FieldValueType *src_ptr = src_field.data().get();
        FieldValueType *dst_ptr = dst_field.data().get();

        bool finished = thrust::reduce(indicator_src_field.begin(), indicator_src_field.end()) == (float)0.0f;
        if (finished) break;

        thrust::for_each(
            thrust::make_counting_iterator<unsigned int>(0),
            thrust::make_counting_iterator<unsigned int>(config.num_evaluation_points),
            [=] __device__(unsigned int idx) {
                // if the node is already visited, skip
                if (indicator_src_ptr[idx] <= winding_number_threshold) {
                    dst_ptr[idx] = src_ptr[idx];
                    indicator_dst_ptr[idx] = 0.0f;
                    return;
                }

                owl::vec_t<int, Dim> idx_md = utils::unflatten(idx, grid_res);

                int num_visted_neighbors = 0;
                FieldValueType value_sum = utils::zero<FieldValueType>();

                if constexpr (Dim == 2) {
                    for (int i : {-1, 1}) {
                        for (int j : {-1, 1}) {
                            owl::vec_t<int, Dim> shift_idx_md = idx_md + owl::vec_t<int, Dim>(i, j);

                            owl::vec_t<int, Dim> shift_idx_md_clamped =
                                owl::min(owl::max(shift_idx_md, utils::zero<owl::vec_t<int, Dim>>()), grid_res - 1);
                            if (shift_idx_md != shift_idx_md_clamped) continue;

                            int shift_idx = utils::flatten(shift_idx_md, grid_res);
                            if (indicator_src_ptr[shift_idx] > winding_number_threshold) continue;

                            value_sum += src_ptr[shift_idx];
                            num_visted_neighbors++;
                        }
                    }
                } else if constexpr (Dim == 3) {
                    for (int i : {-1, 1}) {
                        for (int j : {-1, 1}) {
                            for (int k : {-1, 1}) {
                                owl::vec_t<int, Dim> shift_idx_md = idx_md + owl::vec_t<int, Dim>(i, j, k);

                                owl::vec_t<int, Dim> shift_idx_md_clamped =
                                    owl::min(owl::max(shift_idx_md, utils::zero<owl::vec_t<int, Dim>>()), grid_res - 1);
                                if (shift_idx_md != shift_idx_md_clamped) continue;

                                int shift_idx = utils::flatten(shift_idx_md, grid_res);
                                if (indicator_src_ptr[shift_idx] > winding_number_threshold) continue;

                                value_sum += src_ptr[shift_idx];
                                num_visted_neighbors++;
                            }
                        }
                    }
                }

                if (num_visted_neighbors) {
                    dst_ptr[idx] = value_sum / (float)num_visted_neighbors;
                    indicator_dst_ptr[idx] = 0.0f;
                } else {
                    dst_ptr[idx] = src_ptr[idx];
                    indicator_dst_ptr[idx] = indicator_dst_ptr[idx];
                }
            }
        );

        thrust::swap(indicator_src_field, indicator_dst_field);
        thrust::swap(src_field, dst_field);
    }

    grid_field = src_field;
}

template class Scene<2>;
template class Scene<3>;

template void Scene<2>::advect<float>(
    const thrust::universal_vector<owl::vec_t<float, 2>> &evaluation_point,
    const thrust::universal_vector<owl::vec_t<float, 2>> &velocity, const thrust::universal_vector<float> &field,
    thrust::universal_vector<float> &new_field, const float dt
);
template void Scene<3>::advect<float>(
    const thrust::universal_vector<owl::vec_t<float, 3>> &evaluation_point,
    const thrust::universal_vector<owl::vec_t<float, 3>> &velocity, const thrust::universal_vector<float> &field,
    thrust::universal_vector<float> &new_field, const float dt
);
template void Scene<2>::advect<owl::vec2f>(
    const thrust::universal_vector<owl::vec_t<float, 2>> &evaluation_point,
    const thrust::universal_vector<owl::vec_t<float, 2>> &velocity, const thrust::universal_vector<owl::vec2f> &field,
    thrust::universal_vector<owl::vec2f> &new_field, const float dt
);
template void Scene<3>::advect<owl::vec3f>(
    const thrust::universal_vector<owl::vec_t<float, 3>> &evaluation_point,
    const thrust::universal_vector<owl::vec_t<float, 3>> &velocity, const thrust::universal_vector<owl::vec3f> &field,
    thrust::universal_vector<owl::vec3f> &new_field, const float dt
);

template void Scene<2>::particle_to_grid<float>(
    const thrust::universal_vector<owl::vec_t<float, 2>> &particle_positions,
    const thrust::universal_vector<float> &particle_values, thrust::universal_vector<float> &grid_field,
    thrust::universal_vector<float> &grid_weight_sum, int order
);

template void Scene<3>::particle_to_grid<float>(
    const thrust::universal_vector<owl::vec_t<float, 3>> &particle_positions,
    const thrust::universal_vector<float> &particle_values, thrust::universal_vector<float> &grid_field,
    thrust::universal_vector<float> &grid_weight_sum, int order
);

template void Scene<2>::particle_to_grid<owl::vec2f>(
    const thrust::universal_vector<owl::vec_t<float, 2>> &particle_positions,
    const thrust::universal_vector<owl::vec2f> &particle_values, thrust::universal_vector<owl::vec2f> &grid_field,
    thrust::universal_vector<float> &grid_weight_sum, int order
);

template void Scene<3>::particle_to_grid<owl::vec3f>(
    const thrust::universal_vector<owl::vec_t<float, 3>> &particle_positions,
    const thrust::universal_vector<owl::vec3f> &particle_values, thrust::universal_vector<owl::vec3f> &grid_field,
    thrust::universal_vector<float> &grid_weight_sum, int order
);

template void Scene<2>::grid_breadth_first_fill<owl::vec2f>(
    const thrust::universal_vector<float> &grid_winding_number_buf, thrust::universal_vector<owl::vec2f> &grid_field
);

template void Scene<2>::grid_breadth_first_fill<float>(
    const thrust::universal_vector<float> &grid_winding_number_buf, thrust::universal_vector<float> &grid_field
);

template void Scene<3>::grid_breadth_first_fill<owl::vec3f>(
    const thrust::universal_vector<float> &grid_winding_number_buf, thrust::universal_vector<owl::vec3f> &grid_field
);

template void Scene<3>::grid_breadth_first_fill<float>(
    const thrust::universal_vector<float> &grid_winding_number_buf, thrust::universal_vector<float> &grid_field
);