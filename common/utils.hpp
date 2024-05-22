#pragma once

#include <curand_kernel.h>
#include <fstream>
#include <iomanip>
#include <owl/common/math/vec.h>
#include <random>
#include <thrust/universal_vector.h>

inline std::ostream &operator<<(std::ostream &os, const std::chrono::time_point<std::chrono::system_clock> &time) {
    auto _time = std::chrono::system_clock::to_time_t(time);
    std::tm tm = *std::localtime(&_time);
    // This format is consistent with the format used in __DATE__ and __TIME__ macros.
    os << std::put_time(&tm, "%b %e %Y %H:%M:%S");
    return os;
}

namespace utils {

inline __device__ void random_init(unsigned long long seed, curandStateXORWOW_t &state) {
    curand_init(seed, 0, 0, &state);
}

inline __device__ void random_init(unsigned long long seed, curandStateMRG32k3a_t &state) {
    curand_init(seed, 0, 0, &state);
}

template <class randState_t> void random_states_init(thrust::universal_vector<randState_t> &random_states) {
    std::mt19937_64 random;
    thrust::universal_vector<unsigned long long> random_seeds(random_states.size());
    for (auto &seed : random_seeds) seed = random();
    randState_t *random_states_ptr = random_states.data().get();
    unsigned long long *random_seeds_ptr = random_seeds.data().get();
    thrust::for_each(
        thrust::make_counting_iterator<unsigned int>(0),
        thrust::make_counting_iterator<unsigned int>(random_states.size()),
        [random_states_ptr, random_seeds_ptr] __device__(unsigned int idx) {
            random_init(random_seeds_ptr[idx], random_states_ptr[idx]);
        }
    );
}

inline bool __both__ isfinite(const float &val) { return std::isfinite(val); }
inline bool __both__ isfinite(const owl::vec2f &vec) { return std::isfinite(vec.x) && std::isfinite(vec.y); }
inline bool __both__ isfinite(const owl::vec3f &vec) {
    return std::isfinite(vec.x) && std::isfinite(vec.y) && std::isfinite(vec.z);
}

template <typename T> inline bool isfinite(const thrust::universal_vector<T> &field) {
    for (const auto &elem : field) {
        if (!isfinite(elem)) return false;
    }
    return true;
}

template <typename T, int Dim>
inline void save_field(
    const thrust::universal_vector<T> &field, const std::string &filename, const owl::vec_t<int, Dim> &grid_size
) {
    std::ofstream out(filename.c_str(), std::ios::out | std::ios::binary);
    if (!isfinite(field)) std::cout << "Warinig: field `" << filename << "`is not finite!" << std::endl;

    for (int i = 0; i < Dim; ++i) out.write((const char *)&grid_size[i], sizeof(int));
    out.write((const char *)field.data().get(), sizeof(T) * field.size());
    out.close();
}

template <typename T, int Dim>
inline void
save_field(const std::vector<T> &field, const std::string &filename, const owl::vec_t<int, Dim> &grid_size) {
    std::ofstream out(filename.c_str(), std::ios::out | std::ios::binary);

    for (int i = 0; i < Dim; ++i) out.write((const char *)&grid_size[i], sizeof(int));
    out.write((const char *)field.data(), sizeof(T) * field.size());
    out.close();
}

template <int Dim, typename T> inline void load_field(thrust::universal_vector<T> &field, const std::string &filename) {
    std::ifstream in(filename.c_str(), std::ios::in | std::ios::binary);
    if (!in.is_open()) {
        std::cout << "Failed to open file: " << filename << std::endl;
        exit(1);
    }

    owl::vec_t<int, Dim> grid_size;
    int prod = 1;
    for (int i = 0; i < Dim; ++i) {
        in.read((char *)&grid_size[i], sizeof(int));
        prod *= grid_size[i];
    }
    field.resize(prod);
    in.read((char *)field.data().get(), sizeof(T) * field.size());
    in.close();
}

template <int Dim, typename T> inline void load_field(std::vector<T> &field, const std::string &filename) {
    std::ifstream in(filename.c_str(), std::ios::in | std::ios::binary);
    if (!in.is_open()) {
        std::cout << "Failed to open file: " << filename << std::endl;
        exit(1);
    }

    owl::vec_t<int, Dim> grid_size;
    int prod = 1;
    for (int i = 0; i < Dim; ++i) {
        in.read((char *)&grid_size[i], sizeof(int));
        prod *= grid_size[i];
    }
    field.resize(prod);
    in.read((char *)field.data(), sizeof(T) * field.size());
    in.close();
}

} // namespace utils