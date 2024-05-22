#pragma once

#include <curand_kernel.h>
#include <owl/common/math/vec.h>

namespace utils {

template <typename T> inline __both__ constexpr T zero();
template <> inline __both__ constexpr float zero<float>() { return 0.f; }
template <> inline __both__ owl::vec2f zero<owl::vec2f>() { return {0.f, 0.f}; }
template <> inline __both__ owl::vec3f zero<owl::vec3f>() { return {0.f, 0.f, 0.f}; }
template <> inline __both__ owl::vec2i zero<owl::vec2i>() { return {0, 0}; }
template <> inline __both__ owl::vec3i zero<owl::vec3i>() { return {0, 0, 0}; }

template <typename T> inline __both__ constexpr T one();
template <> inline __both__ constexpr float one<float>() { return 1.f; }
template <> inline __both__ owl::vec2f one<owl::vec2f>() { return {1.f, 1.f}; }
template <> inline __both__ owl::vec3f one<owl::vec3f>() { return {1.f, 1.f, 1.f}; }
template <> inline __both__ owl::vec2i one<owl::vec2i>() { return {1, 1}; }
template <> inline __both__ owl::vec3i one<owl::vec3i>() { return {1, 1, 1}; }

using randState_t = curandStateXORWOW_t;
// using randState_t = curandStateMRG32k3a_t;

inline __device__ float rand_uniform(curandStateXORWOW_t &random_state) noexcept {
    return curand_uniform(&random_state);
}
inline __device__ float rand_normal(curandStateXORWOW_t &random_state) noexcept { return curand_normal(&random_state); }

inline __device__ float rand_uniform(curandStateMRG32k3a_t &random_state) noexcept {
    return curand_uniform(&random_state);
}
inline __device__ float rand_normal(curandStateMRG32k3a_t &random_state) noexcept {
    return curand_normal(&random_state);
}

inline __both__ owl::vec_t<int, 3> unflatten(int idx, const owl::vec_t<int, 3> &grid_res) noexcept {
    return owl::vec_t<int, 3>(idx % grid_res.x, (idx / grid_res.x) % grid_res.y, idx / (grid_res.x * grid_res.y));
}
inline __both__ owl::vec_t<int, 2> unflatten(int idx, const owl::vec_t<int, 2> &grid_res) noexcept {
    return owl::vec_t<int, 2>(idx % grid_res.x, idx / grid_res.x);
}

inline __both__ int flatten(const owl::vec_t<int, 2> &idx, const owl::vec_t<int, 2> &grid_res) noexcept {
    return idx.y * grid_res.x + idx.x;
}

inline __both__ int flatten(const owl::vec_t<int, 3> &idx, const owl::vec_t<int, 3> &grid_res) noexcept {
    return idx.z * grid_res.y * grid_res.x + idx.y * grid_res.x + idx.x;
}

// returns a point in (-domain_size/2, domain_size/2)^Dim
template <int Dim>
inline __both__ owl::vec_t<float, Dim>
idx_to_domain_point(int idx, const owl::vec_t<int, Dim> &grid_res, const owl::vec_t<float, Dim> &domain_size) noexcept {
    const owl::vec_t<float, Dim> dx = domain_size / owl::vec_t<float, Dim>(grid_res);
    owl::vec_t<int, Dim> x = unflatten(idx, grid_res);
    return (owl::vec_t<float, Dim>(x) + 0.5f * one<owl::vec_t<float, Dim>>()) * dx - 0.5f * domain_size;
}

// returns a point in (-domain_size/2, domain_size/2)^Dim
template <int Dim>
inline __both__ owl::vec_t<float, Dim> idx_to_domain_point(
    const owl::vec_t<int, Dim> &idx, const owl::vec_t<int, Dim> &grid_res, const owl::vec_t<float, Dim> &domain_size
) noexcept {
    const owl::vec_t<float, Dim> dx = domain_size / owl::vec_t<float, Dim>(grid_res);
    return (owl::vec_t<float, Dim>(idx) + 0.5f * one<owl::vec_t<float, Dim>>()) * dx - 0.5f * domain_size;
}

// maps (-domain_size/2, domain_size/2)^Dim to Dim-dimensional idx.
// Note (-domain_size/2, -domain_size/2) maps to (-0.5, -0.5) instead of (0, 0).
template <int Dim>
inline __both__ owl::vec_t<float, Dim> domain_point_to_idx(
    const owl::vec_t<float, Dim> &p, const owl::vec_t<int, Dim> &grid_res, const owl::vec_t<float, Dim> &domain_size
) noexcept {
    return (owl::vec_t<float, Dim>(grid_res) / domain_size) * p + .5f * owl::vec_t<float, Dim>(grid_res) -
           .5f * one<owl::vec_t<float, Dim>>();
}

// Similar to OpenGL texture wrap mode
enum class WrapMode : int { Repeat, ClampToEdge, ClampToBorder };

inline __both__ float lerp(const float v0, const float v1, const float t) { return fma(t, v1, fma(-t, v0, v0)); }
inline __both__ owl::vec2f lerp(const owl::vec2f &v0, const owl::vec2f &v1, float t) {
    return {lerp(v0.x, v1.x, t), lerp(v0.y, v1.y, t)};
}
inline __both__ owl::vec3f lerp(const owl::vec3f &v0, const owl::vec3f &v1, float t) {
    return {lerp(v0.x, v1.x, t), lerp(v0.y, v1.y, t), lerp(v0.z, v1.z, t)};
}

template <typename T>
inline __both__ T sample_buffer_linear(
    const owl::vec_t<float, 2> &idx, const T *buffer, const owl::vec_t<int, 2> &grid_res,
    const WrapMode wrap_mode = WrapMode::ClampToBorder, const T &clamp_value = zero<T>()
) {
    owl::vec_t<int, 2> floor_idx(floorf(idx.x), floorf(idx.y));

    T cache_val[2][2];

    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++) {
            owl::vec2i int_idx = floor_idx + owl::vec2i(i, j);
            if (wrap_mode == WrapMode::Repeat) {
                int_idx = owl::vec2i(int_idx.x % grid_res.x, int_idx.y % grid_res.y);
                int_idx += grid_res;
                int_idx = owl::vec2i(int_idx.x % grid_res.x, int_idx.y % grid_res.y);
            } else if (wrap_mode == WrapMode::ClampToEdge)
                int_idx =
                    owl::vec2i(owl::clamp(int_idx.x, 0, grid_res.x - 1), (owl::clamp(int_idx.y, 0, grid_res.y - 1)));
            else if (wrap_mode == WrapMode::ClampToBorder) {
                owl::vec2i int_idx2(
                    owl::clamp(int_idx.x, 0, grid_res.x - 1), (owl::clamp(int_idx.y, 0, grid_res.y - 1))
                );
                if (int_idx != int_idx2) {
                    cache_val[i][j] = clamp_value;
                    continue;
                }
            }
            cache_val[i][j] = buffer[flatten(int_idx, grid_res)];
        }

    return lerp(
        lerp(cache_val[0][0], cache_val[0][1], idx.y - floor_idx.y),
        lerp(cache_val[1][0], cache_val[1][1], idx.y - floor_idx.y), idx.x - floor_idx.x
    );
}

template <typename T>
inline __both__ T sample_buffer_linear(
    const owl::vec_t<float, 3> &idx, const T *buffer, const owl::vec_t<int, 3> &grid_res,
    const WrapMode wrap_mode = WrapMode::ClampToBorder, const T &clamp_value = zero<T>()
) {
    owl::vec_t<int, 3> floor_idx(floorf(idx.x), floorf(idx.y), floorf(idx.z));

    T cache_val[2][2][2];

    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++) {
                owl::vec_t<int, 3> int_idx = floor_idx + owl::vec3i(i, j, k);
                if (wrap_mode == WrapMode::Repeat) {
                    int_idx = owl::vec3i(int_idx.x % grid_res.x, int_idx.y % grid_res.y, int_idx.z % grid_res.z);
                    int_idx += grid_res;
                    int_idx = owl::vec3i(int_idx.x % grid_res.x, int_idx.y % grid_res.y, int_idx.z % grid_res.z);
                } else if (wrap_mode == WrapMode::ClampToEdge)
                    int_idx = owl::vec3i(
                        owl::clamp(int_idx.x, 0, grid_res.x - 1), (owl::clamp(int_idx.y, 0, grid_res.y - 1)),
                        owl::clamp(int_idx.z, 0, grid_res.z - 1)
                    );
                else if (wrap_mode == WrapMode::ClampToBorder) {
                    owl::vec3i int_idx2(
                        owl::clamp(int_idx.x, 0, grid_res.x - 1), (owl::clamp(int_idx.y, 0, grid_res.y - 1)),
                        owl::clamp(int_idx.z, 0, grid_res.z - 1)
                    );
                    if (int_idx != int_idx2) {
                        cache_val[i][j][k] = clamp_value;
                        continue;
                    }
                }
                cache_val[i][j][k] = buffer[flatten(int_idx, grid_res)];
            }

    return lerp(
        lerp(
            lerp(cache_val[0][0][0], cache_val[0][0][1], idx.z - floor_idx.z),
            lerp(cache_val[0][1][0], cache_val[0][1][1], idx.z - floor_idx.z), idx.y - floor_idx.y
        ),
        lerp(
            lerp(cache_val[1][0][0], cache_val[1][0][1], idx.z - floor_idx.z),
            lerp(cache_val[1][1][0], cache_val[1][1][1], idx.z - floor_idx.z), idx.y - floor_idx.y
        ),
        idx.x - floor_idx.x
    );
}

// Compensated sum algorithm by Kahan
template <typename T> struct KahanSum {
    T sum, c;
    inline __both__ KahanSum() {
        sum = zero<T>();
        c = zero<T>();
    };
    inline __both__ KahanSum(const KahanSum &other) {
        this->sum = other.sum;
        this->c = other.c;
    }
    inline __both__ KahanSum<T> &operator+=(const T &value) {
        T y = value - c;
        T t = sum + y;
        c = (t - sum) - y;
        sum = t;
        return *this;
    };
    inline __both__ KahanSum<T> &operator-=(const T &value) {
        T y = -value - c;
        T t = sum + y;
        c = (t - sum) - y;
        sum = t;
        return *this;
    };
    inline __both__ operator T() const { return sum; }
};

} // namespace utils

// extensions to the vector class in owl
namespace owl {
inline bool __both__ isfinite(const float &val) { return std::isfinite(val); }
inline bool __both__ isfinite(const owl::vec2f &vec) { return std::isfinite(vec.x) && std::isfinite(vec.y); }
inline bool __both__ isfinite(const owl::vec3f &vec) {
    return std::isfinite(vec.x) && std::isfinite(vec.y) && std::isfinite(vec.z);
}

template <typename T, int Dim> inline __both__ owl::vec_t<T, Dim> normalize(const owl::vec_t<T, Dim> &v) {
    T divisor = owl::common::polymorphic::rsqrt(dot(v, v));
    if (!isfinite(divisor)) return utils::zero<owl::vec_t<T, Dim>>();
    return v * divisor;
}

template <typename T, int Dim> inline __both__ T length(const owl::vec_t<T, Dim> &v) {
    return owl::common::polymorphic::sqrt(dot(v, v));
}

template <typename T, int Dim> inline __both__ T length2(const owl::vec_t<T, Dim> &v) { return dot(v, v); }

template <int Dim> inline __both__ owl::vec_t<float, Dim> cast_dim(const owl::vec4f &vec);
template <> inline __both__ owl::vec2f cast_dim<2>(const owl::vec4f &vec) { return {vec.x, vec.y}; }
template <> inline __both__ owl::vec3f cast_dim<3>(const owl::vec4f &vec) { return {vec.x, vec.y, vec.z}; }
template <> inline __both__ owl::vec4f cast_dim<4>(const owl::vec4f &vec) { return vec; }

template <int Dim> inline __both__ owl::vec_t<float, Dim> cast_dim(const owl::vec3f &vec);
template <> inline __both__ owl::vec2f cast_dim<2>(const owl::vec3f &vec) { return {vec.x, vec.y}; }
template <> inline __both__ owl::vec3f cast_dim<3>(const owl::vec3f &vec) { return vec; }
template <> inline __both__ owl::vec4f cast_dim<4>(const owl::vec3f &vec) { return {vec.x, vec.y, vec.z, 0.f}; }

template <int Dim> inline __both__ owl::vec_t<float, Dim> cast_dim(const owl::vec2f &vec);
template <> inline __both__ owl::vec2f cast_dim<2>(const owl::vec2f &vec) { return vec; }
template <> inline __both__ owl::vec3f cast_dim<3>(const owl::vec2f &vec) { return {vec.x, vec.y, 0.f}; }
template <> inline __both__ owl::vec4f cast_dim<4>(const owl::vec2f &vec) { return {vec.x, vec.y, 0.f, 0.f}; }

} // namespace owl