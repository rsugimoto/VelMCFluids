#pragma once

#include "common.cuh"
#include <cstring>
#include <optix_device.h>
#include <owl/common/math/AffineSpace.h>
#include <owl/owl.h>

enum class DomainType : int { BoundedDomain, UnboundedDomain };
enum class AdvectionMode : int { Euler, MacCormack, RK3, RK4 };

inline const char *domainTypeToString(const DomainType domain_type) {
    switch (domain_type) {
    case DomainType::BoundedDomain: return "BoundedDomain";
    case DomainType::UnboundedDomain: return "UnboundedDomain";
    default: throw std::runtime_error("Invalid DomainType");
    }
}

inline DomainType stringToDomainType(const char *string) {
    if (strcmp(string, "BoundedDomain") == 0) return DomainType::BoundedDomain;
    if (strcmp(string, "UnboundedDomain") == 0) return DomainType::UnboundedDomain;
    throw std::runtime_error("Invalid DomainType");
}

inline const char *advectionModeToString(const AdvectionMode advection_mode) {
    switch (advection_mode) {
    case AdvectionMode::Euler: return "Euler";
    case AdvectionMode::MacCormack: return "MacCormack";
    case AdvectionMode::RK3: return "RK3";
    case AdvectionMode::RK4: return "RK4";
    default: throw std::runtime_error("Invalid AdvectionMode");
    }
}

inline AdvectionMode stringToAdvectionMode(const char *string) {
    if (strcmp(string, "Euler") == 0) return AdvectionMode::Euler;
    if (strcmp(string, "RK1") == 0) return AdvectionMode::Euler;
    if (strcmp(string, "MacCormack") == 0) return AdvectionMode::MacCormack;
    if (strcmp(string, "RK2") == 0) return AdvectionMode::MacCormack;
    if (strcmp(string, "RK3") == 0) return AdvectionMode::RK3;
    if (strcmp(string, "RK4") == 0) return AdvectionMode::RK4;
    throw std::runtime_error("Invalid AdvectionMode");
}

struct Mesh {
    owl::vec3f *vertex_buf;
    owl::vec3i *index_buf;
    owl::vec3f *normal_buf;
    owl::affine3f transform;
    float *area_buf;
    float *cdf_buf;
    int num_primitives;
};

template <int Dim> struct DeviceConfig {
    int path_length;
    int num_path_samples;
    int num_volume_samples_indirect;
    int num_pseudo_boundary_samples_indirect;
    int num_volume_samples_direct;
    int num_pseudo_boundary_samples_direct;
    int num_winding_samples;
    int num_init_winding_samples;
    owl::vec_t<int, Dim> grid_res;
    owl::vec_t<float, Dim> domain_size;
    float dGdx_regularization;
    DomainType domain_type;
    AdvectionMode advection_mode;
    bool enable_antithetic_sampling; // for volume term
};

template <int Dim> struct WindingNumberRayGenData {
    OptixTraversableHandle acc_structure;
    DeviceConfig<Dim> config;
    int num_primitives;
    int device_index;
    int device_count;
    owl::vec_t<float, Dim> *evaluation_point_buf;
    float *winding_number_buf;
    utils::randState_t *random_state_buf;
    int num_evaluation_points;
};

enum class RayType : int { IntersectionSamplingRay = 0, WindingNumberRay, ClosestBoundaryPointRay, NumRayTypes };

template <int Dim> struct VolumeSample {
    owl::vec_t<float, Dim> position;
    float inv_pdf;
};

template <int Dim> struct BoundaryPoint {
    owl::vec_t<float, Dim> position;
    int prim_id;
};

template <int Dim> struct BoundarySample {
    BoundaryPoint<Dim> boundary_point;
    float inv_pdf;
};

template <int Dim> struct RayIntersectionBoundarySample {
    BoundaryPoint<Dim> boundary_point;
    int num_intersections;
};

template <int Dim> inline __device__ owl::vec_t<float, Dim> uniform_direction_sample(utils::randState_t &random);
template <int Dim>
inline __device__ VolumeSample<Dim> strongly_singular_ball_sample(float radius, utils::randState_t &random);

template <int Dim> __device__ BoundarySample<Dim> cdf_boundary_sample(const Mesh &mesh, utils::randState_t &random);

// This is callable from OptiX programs only
template <int Dim>
__device__ RayIntersectionBoundarySample<Dim> line_intersection_boundary_sample(
    OptixTraversableHandle acc_structure, const int origin_prim_id, const owl::vec_t<float, Dim> &origin,
    utils::randState_t &random
);

// This is callable from OptiX programs only
template <int Dim>
__device__ RayIntersectionBoundarySample<Dim> line_intersection_boundary_sample(
    OptixTraversableHandle acc_structure, const owl::vec_t<float, Dim> &origin, utils::randState_t &random
);

// This is callable from OptiX programs only
template <int Dim>
__device__ RayIntersectionBoundarySample<Dim> line_intersection_boundary_sample(
    OptixTraversableHandle acc_structure, const BoundarySample<Dim> &origin, utils::randState_t &random
);

// This is callable from OptiX programs only
template <int Dim>
__device__ int line_intersection_count(
    const OptixTraversableHandle acc_structure, const int origin_prim_id, const owl::vec_t<float, Dim> &origin,
    const owl::vec_t<float, Dim> &dir
);

// This is callable from OptiX programs only
template <int Dim>
__device__ int line_intersection_count(
    OptixTraversableHandle acc_structure, const owl::vec_t<float, Dim> &origin, const owl::vec_t<float, Dim> &dir
);

// This is callable from OptiX programs only
template <int Dim>
__device__ int line_intersection_count(
    OptixTraversableHandle acc_structure, const BoundaryPoint<Dim> &origin, const owl::vec_t<float, Dim> &dir
);
// This is callable from OptiX programs only
template <int Dim>
__device__ float winding_number_estimator(
    OptixTraversableHandle acc_structure, const owl::vec_t<float, Dim> &query_point, utils::randState_t &random
);

template <int Dim>
__device__ float winding_number_estimator(
    OptixTraversableHandle acc_structure, const owl::vec_t<float, Dim> &query_point, utils::randState_t &random,
    int num_samples
);

template <int Dim>
__device__ float volume_sample_integral_multiplier(
    OptixTraversableHandle acc_structure, const owl::vec_t<float, Dim> &query_point, utils::randState_t &random,
    int num_samples, DomainType domain_type
);

// This is callable from OptiX programs only.
template <int Dim>
__device__ BoundaryPoint<Dim> closest_boundary_point(
    const OptixTraversableHandle acc_structure, const int origin_prim_id, const owl::vec_t<float, Dim> &query_point,
    const owl::vec_t<float, Dim> &line_origin, const owl::vec_t<float, Dim> &line_dir, float ray_t_max = owl::infty(),
    bool use_ray = false
);

// This is callable from OptiX programs only.
template <int Dim>
__device__ BoundaryPoint<Dim> closest_boundary_point(
    const OptixTraversableHandle acc_structure, const owl::vec_t<float, Dim> &query_point,
    const owl::vec_t<float, Dim> &line_origin, const owl::vec_t<float, Dim> &line_dir, float ray_t_max = owl::infty(),
    bool use_ray = false
);

// This is callable from OptiX programs only.
template <int Dim>
__device__ BoundaryPoint<Dim> closest_boundary_point(
    const OptixTraversableHandle acc_structure, const BoundaryPoint<Dim> &qury_point,
    const owl::vec_t<float, Dim> &line_origin, const owl::vec_t<float, Dim> &line_dir, float ray_t_max = owl::infty(),
    bool use_ray = false
);

// This implements a linear search for closest point query.
inline __device__ BoundaryPoint<3> closest_point_in_mesh(Mesh &mesh, const owl::vec3f &query_point);
inline __device__ BoundaryPoint<2> closest_point_in_mesh(Mesh &mesh, const owl::vec2f &query_point);

/* Implementation of the functions declared above */

template <> inline __device__ owl::vec3f uniform_direction_sample<3>(utils::randState_t &rand_state) {
    float phi = 2.f * M_PIf32 * utils::rand_uniform(rand_state);
    float z = 2.f * utils::rand_uniform(rand_state) - 1.f;
    float x = cosf(phi) * sqrtf(1.f - z * z);
    float y = sinf(phi) * sqrtf(1.f - z * z);
    return {x, y, z};
}

template <> inline __device__ owl::vec2f uniform_direction_sample<2>(utils::randState_t &rand_state) {
    float theta = 2.f * M_PIf32 * utils::rand_uniform(rand_state);
    float x = cosf(theta);
    float y = sinf(theta);
    return {x, y};
}

template <>
inline __device__ VolumeSample<3> strongly_singular_ball_sample<3>(float radius, utils::randState_t &rand_state) {
    float r = radius * utils::rand_uniform(rand_state);
    const owl::vec3f dir = uniform_direction_sample<3>(rand_state);
    float inv_pdf = 4.f * M_PIf32 * radius * r * r;
    return {r * dir, inv_pdf};
};

template <>
inline __device__ VolumeSample<2> strongly_singular_ball_sample<2>(float radius, utils::randState_t &rand_state) {
    float r = radius * utils::rand_uniform(rand_state);
    const owl::vec2f dir = uniform_direction_sample<2>(rand_state);
    float inv_pdf = 2.f * M_PIf32 * r * radius;
    return {r * dir, inv_pdf};
}

inline __device__ int binary_search(const float *array, int array_size, float key) {
    int low = 0, high = array_size - 1;
    while (low < high) {
        int mid = (high + low) / 2;
        if (1.f - key <= array[mid]) {
            high = mid;
        } else {
            low = mid + 1;
        }
    }
    return high;
}

inline __device__ owl::vec2f
uniform_sample_point_in_line(const owl::vec2f &v0, const owl::vec2f &v1, utils::randState_t &rand_state) {
    float unif = utils::rand_uniform(rand_state);
    return (1.f - unif) * v0 + unif * v1;
}

inline __device__ owl::vec3f uniform_sample_point_in_triangle(
    const owl::vec3f &v0, const owl::vec3f &v1, const owl::vec3f &v2, utils::randState_t &rand_state
) {
    float unif1 = utils::rand_uniform(rand_state);
    float unif2 = utils::rand_uniform(rand_state);
    float sqrt_unif1 = std::sqrt(unif1);
    return (1.f - sqrt_unif1) * v0 + sqrt_unif1 * (1.f - unif2) * v1 + unif2 * sqrt_unif1 * v2;
}

template <int Dim>
__device__ BoundarySample<Dim> cdf_boundary_sample(const Mesh &mesh, utils::randState_t &rand_state) {
    const unsigned int num_primitives = mesh.num_primitives;
    if (!num_primitives) return BoundarySample<Dim>{BoundaryPoint<Dim>{}, 0.f};

    float unif = utils::rand_uniform(rand_state);
    const int prim_id = binary_search(mesh.cdf_buf, num_primitives, unif);
    const owl::vec3i index = mesh.index_buf[prim_id];
    const float area = mesh.area_buf[prim_id];
    float pmf = prim_id == 0 ? mesh.cdf_buf[0] : mesh.cdf_buf[prim_id] - mesh.cdf_buf[prim_id - 1];

    BoundarySample<Dim> sample;
    sample.boundary_point.prim_id = prim_id;
    if constexpr (Dim == 2)
        sample.boundary_point.position = uniform_sample_point_in_line(
            owl::vec2f(mesh.vertex_buf[index.x].x, mesh.vertex_buf[index.x].y),
            owl::vec2f(mesh.vertex_buf[index.y].x, mesh.vertex_buf[index.y].y), rand_state
        );
    if constexpr (Dim == 3)
        sample.boundary_point.position = uniform_sample_point_in_triangle(
            mesh.vertex_buf[index.x], mesh.vertex_buf[index.y], mesh.vertex_buf[index.z], rand_state
        );
    sample.boundary_point.position =
        owl::cast_dim<Dim>(owl::xfmPoint(mesh.transform, owl::cast_dim<3>(sample.boundary_point.position)));
    sample.inv_pdf = area / pmf;
    return sample;
}

template <int Dim> inline __device__ owl::vec_t<float, Dim> uniform_sample_dir(utils::randState_t &rand_state);
template <> inline __device__ owl::vec2f uniform_sample_dir<2>(utils::randState_t &rand_state) {
    float phi = 2.f * M_PIf32 * utils::rand_uniform(rand_state);
    return owl::vec2f(cosf(phi), sinf(phi));
}
template <> inline __device__ owl::vec3f uniform_sample_dir<3>(utils::randState_t &rand_state) {
    float z = 1.f - 2.f * utils::rand_uniform(rand_state);
    float r = sqrtf(max(1.f - z * z, 0.f));
    float phi = 2.f * M_PIf32 * utils::rand_uniform(rand_state);
    return owl::vec3f(r * cosf(phi), r * sinf(phi), z);
}

template <int Dim>
__device__ float winding_number_estimator(
    const OptixTraversableHandle acc_structure, const owl::vec_t<float, Dim> &query_point,
    utils::randState_t &rand_state
) {
    const owl::vec_t<float, Dim> dir = uniform_sample_dir<Dim>(rand_state);
    owl::RayT<(int)RayType::WindingNumberRay, (int)RayType::NumRayTypes> ray(
        owl::cast_dim<3>(query_point), owl::cast_dim<3>(dir), 0.0f, owl::infty()
    );
    float winding_number_estimate = 0.f;
    owl::traceRay(acc_structure, ray, winding_number_estimate);
    return winding_number_estimate;
}

template <int Dim>
__device__ float winding_number_estimator(
    OptixTraversableHandle acc_structure, const owl::vec_t<float, Dim> &query_point, utils::randState_t &random,
    int num_samples
) {
    if (num_samples == 0) return 0.0f;
    float winding_number_estimate = 0.f;
    for (int i = 0; i < num_samples; ++i) {
        winding_number_estimate += winding_number_estimator(acc_structure, query_point, random);
    }
    return winding_number_estimate / num_samples;
}

template <int Dim>
__device__ float volume_integral_multiplier(
    OptixTraversableHandle acc_structure, const owl::vec_t<float, Dim> &query_point, utils::randState_t &random,
    int num_samples, int num_primitives, DomainType domain_type
) {
    if (num_primitives <= 0) return 1.f;
    float winding_number = winding_number_estimator(acc_structure, query_point, random, num_samples);
    return domain_type == DomainType::BoundedDomain ? winding_number : 1.f + winding_number;
}

struct RayIntersectionSamplingPRD {
    int origin_prim_id;
    int num_intersections;
    float random_sample;
    owl::vec3f position;
    int prim_id;
};

template <int Dim>
__device__ RayIntersectionBoundarySample<Dim> line_intersection_boundary_sample(
    const OptixTraversableHandle acc_structure, const int origin_prim_id, const owl::vec_t<float, Dim> &origin,
    utils::randState_t &rand_state
) {
    const owl::vec_t<float, Dim> dir = uniform_sample_dir<Dim>(rand_state);
    owl::RayT<(int)RayType::IntersectionSamplingRay, (int)RayType::NumRayTypes> ray(
        owl::cast_dim<3>(origin), owl::cast_dim<3>(dir), 0.0f, owl::infty()
    );
    RayIntersectionSamplingPRD prd{origin_prim_id, 0, utils::rand_uniform(rand_state), {0}, 0};
    owl::traceRay(acc_structure, ray, prd);
    ray.direction = -ray.direction;
    owl::traceRay(acc_structure, ray, prd);
    return {
        BoundaryPoint<Dim>{owl::cast_dim<Dim>(owl::vec2f(prd.position.x, prd.position.y)), prd.prim_id},
        prd.num_intersections};
}

// This is callable from OptiX programs only
template <int Dim>
__device__ RayIntersectionBoundarySample<Dim> line_intersection_boundary_sample(
    OptixTraversableHandle acc_structure, const owl::vec_t<float, Dim> &origin, utils::randState_t &random
) {
    return line_intersection_boundary_sample(acc_structure, -1, origin, random);
}

// This is callable from OptiX programs only
template <int Dim>
__device__ RayIntersectionBoundarySample<Dim> line_intersection_boundary_sample(
    OptixTraversableHandle acc_structure, const BoundaryPoint<Dim> &origin, utils::randState_t &random
) {
    return line_intersection_boundary_sample(acc_structure, origin.prim_id, origin.position, random);
}

struct ClosestBoundaryPointPRD {
    int origin_prim_id;
    int num_intersections;
    owl::vec3f query_point;
    float dist2;
    owl::vec3f position;
    int prim_id;
};

template <int Dim>
__device__ BoundaryPoint<Dim> closest_boundary_point(
    const OptixTraversableHandle acc_structure, const int origin_prim_id, const owl::vec_t<float, Dim> &query_point,
    const owl::vec_t<float, Dim> &line_origin, const owl::vec_t<float, Dim> &line_dir, float ray_t_max, bool use_ray
) {
    owl::RayT<(int)RayType::ClosestBoundaryPointRay, (int)RayType::NumRayTypes> ray(
        owl::cast_dim<3>(line_origin), owl::cast_dim<3>(line_dir), 0.0f, ray_t_max
    );
    ClosestBoundaryPointPRD prd{origin_prim_id, 0, owl::cast_dim<3>(query_point), owl::infty(), {0}, -1};
    owl::traceRay(acc_structure, ray, prd);
    if (!use_ray) {
        ray.direction = -ray.direction;
        owl::traceRay(acc_structure, ray, prd);
    }
    return {owl::cast_dim<Dim>(owl::vec2f(prd.position.x, prd.position.y)), prd.prim_id};
}

template <int Dim>
__device__ BoundaryPoint<Dim> closest_boundary_point(
    const OptixTraversableHandle acc_structure, const owl::vec_t<float, Dim> &query_point,
    const owl::vec_t<float, Dim> &line_origin, const owl::vec_t<float, Dim> &line_dir, float ray_t_max, bool use_ray
) {
    return closest_boundary_point(acc_structure, -1, query_point, line_origin, line_dir, ray_t_max, use_ray);
}

template <int Dim>
__device__ BoundaryPoint<Dim> closest_boundary_point(
    const OptixTraversableHandle acc_structure, const BoundaryPoint<Dim> &query_point,
    const owl::vec_t<float, Dim> &line_origin, const owl::vec_t<float, Dim> &line_dir, float ray_t_max, bool use_ray
) {
    return closest_boundary_point(
        acc_structure, query_point.prim_id, query_point.position, line_origin, line_dir, ray_t_max, use_ray
    );
}

template <int Dim>
__device__ int line_intersection_count(
    const OptixTraversableHandle acc_structure, const int origin_prim_id, const owl::vec_t<float, Dim> &origin,
    const owl::vec_t<float, Dim> &dir
) {
    owl::RayT<(int)RayType::ClosestBoundaryPointRay, (int)RayType::NumRayTypes> ray(
        owl::cast_dim<3>(origin), owl::cast_dim<3>(dir), 0.0f, owl::infty()
    );
    ClosestBoundaryPointPRD prd{origin_prim_id, 0, {0}, 0, {0}, -1};
    owl::traceRay(acc_structure, ray, prd);
    ray.direction = -ray.direction;
    owl::traceRay(acc_structure, ray, prd);
    return prd.num_intersections;
}

template <int Dim>
__device__ int line_intersection_count(
    OptixTraversableHandle acc_structure, const owl::vec_t<float, Dim> &origin, const owl::vec_t<float, Dim> &dir
) {
    return line_intersection_count(acc_structure, -1, origin, dir);
}

template <int Dim>
__device__ int line_intersection_count(
    OptixTraversableHandle acc_structure, const BoundaryPoint<Dim> &origin, const owl::vec_t<float, Dim> &dir
) {
    return line_intersection_count(acc_structure, origin.prim_id, origin.position, dir);
}

// based on  https://github.com/gszauer/GamePhysicsCookbook/blob/master/Code/Geometry3D.cpp

using Point = owl::vec3f;
struct Line {
    Point start;
    Point end;
};
struct Plane {
    owl::vec3f normal;
    float distance;
};
struct Triangle {
    owl::vec3f a, b, c;
};

inline __device__ Point closest_point_in_plane(const Plane &plane, const Point &point) noexcept {
    float distance = dot(plane.normal, point) - plane.distance;
    return point - plane.normal * distance;
}

inline __device__ Point closest_point_in_line(const Line &line, const Point &point) noexcept {
    owl::vec3f lVec = line.end - line.start; // Line Vector
    float t = dot(point - line.start, lVec) / owl::length2(lVec);
    t = min(max(t, 0.0f), 1.0f);
    return line.start + lVec * t;
}

inline __device__ Plane plane_from_triangle(const Triangle &t) noexcept {
    Plane result;
    result.normal = normalize(cross(t.b - t.a, t.c - t.a));
    result.distance = dot(result.normal, t.a);
    return result;
}

inline __device__ bool point_in_triangle(const Point &p, const Triangle &t) noexcept {
    owl::vec3f a = t.a - p;
    owl::vec3f b = t.b - p;
    owl::vec3f c = t.c - p;

    owl::vec3f normPBC = cross(b, c); // Normal of PBC (u)
    owl::vec3f normPCA = cross(c, a); // Normal of PCA (v)
    owl::vec3f normPAB = cross(a, b); // Normal of PAB (w)

    if (dot(normPBC, normPCA) < 0.0f)
        return false;
    else if (dot(normPBC, normPAB) < 0.0f)
        return false;

    return true;
}

inline __device__ owl::vec3f closest_point_in_triangle(const owl::vec3f &p, const Triangle &t) noexcept {
    Plane plane = plane_from_triangle(t);
    Point closest = closest_point_in_plane(plane, p);

    // Closest point was inside triangle
    if (point_in_triangle(closest, t)) { return closest; }

    Point c1 = closest_point_in_line(Line{t.a, t.b}, closest); // Line AB
    Point c2 = closest_point_in_line(Line{t.b, t.c}, closest); // Line BC
    Point c3 = closest_point_in_line(Line{t.c, t.a}, closest); // Line CA

    float magSq1 = owl::length2(closest - c1);
    float magSq2 = owl::length2(closest - c2);
    float magSq3 = owl::length2(closest - c3);

    if (magSq1 <= magSq2 && magSq1 <= magSq3) {
        return c1;
    } else if (magSq2 <= magSq1 && magSq2 <= magSq3) {
        return c2;
    }

    return c3;
}

inline __device__ owl::vec3f closest_point_in_triangle(
    const owl::vec3f &query_point, const owl::vec3f &v0, const owl::vec3f &v1, const owl::vec3f &v2
) noexcept {
    return closest_point_in_triangle(query_point, Triangle{v0, v1, v2});
}

inline __device__ BoundaryPoint<3> closest_point_in_mesh(const Mesh &mesh, const owl::vec3f &query_point) {
    owl::affine3f inv_transform = rcp(mesh.transform);
    owl::vec3f query_point_local = owl::xfmPoint(inv_transform, query_point);
    float min_dist = INFINITY;
    int closest_prim_id = -1;
    owl::vec3f closest_point;
    for (int i = 0; i < mesh.num_primitives; ++i) {
        const owl::vec3i index = mesh.index_buf[i];
        const owl::vec3f v0 = mesh.vertex_buf[index.x];
        const owl::vec3f v1 = mesh.vertex_buf[index.y];
        const owl::vec3f v2 = mesh.vertex_buf[index.z];
        const owl::vec3f closest_point_on_triangle = closest_point_in_triangle(query_point_local, v0, v1, v2);
        const float dist = owl::length(closest_point_on_triangle - query_point_local);
        if (dist < min_dist) {
            min_dist = dist;
            closest_prim_id = i;
            closest_point = closest_point_on_triangle;
        }
    }
    return {owl::xfmPoint(mesh.transform, closest_point), closest_prim_id};
}

inline __device__ BoundaryPoint<2> closest_point_in_mesh(const Mesh &mesh, const owl::vec2f &query_point) {
    BoundaryPoint<3> point_3d = closest_point_in_mesh(mesh, owl::cast_dim<3>(query_point));
    return {owl::cast_dim<2>(point_3d.position), point_3d.prim_id};
}