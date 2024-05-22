#include <geometry.cuh>

OPTIX_ANY_HIT_PROGRAM(WindingNumberAnyHit)() {
    float &estimate = owl::getPRD<float>();
    const owl::vec3f dir = optixGetWorldRayDirection();
    const owl::vec3f *normal_buf = owl::getProgramData<const owl::vec3f *>();
    const int prim_id = optixGetPrimitiveIndex();
    const owl::vec3f n = optixTransformNormalFromObjectToWorldSpace(normal_buf[prim_id]);
    estimate += owl::dot(n, dir) > 0.0f ? 1.0f : -1.0f;
    optixIgnoreIntersection();
}

OPTIX_ANY_HIT_PROGRAM(RayIntersectionSamplingAnyHit)() {
    RayIntersectionSamplingPRD &prd = owl::getPRD<RayIntersectionSamplingPRD>();
    const int prim_id = optixGetPrimitiveIndex();
    if (prd.origin_prim_id == prim_id) {
        optixIgnoreIntersection();
        return;
    }

    prd.num_intersections++;

    // Chao’s algorithm with explicit warping
    // as summarized in Ogaki 2021
    // https:dl.acm.org/doi/10.1145/3478512.3488602

    float p = 1.f / prd.num_intersections;
    if (prd.random_sample < p) {
        const owl::vec3f ray_orgin = optixGetWorldRayOrigin();
        const owl::vec3f ray_dir = optixGetWorldRayDirection();
        const float hit_t = optixGetRayTmax();
        const owl::vec3f hit_pos = ray_orgin + hit_t * ray_dir;

        prd.prim_id = prim_id;
        prd.position = hit_pos;

        prd.random_sample /= p;
    } else {
        prd.random_sample = (prd.random_sample - p) / (1.f - p);
    }

    optixIgnoreIntersection();
}

OPTIX_ANY_HIT_PROGRAM(ClosestBoundaryPointAnyHit)() {
    ClosestBoundaryPointPRD &prd = owl::getPRD<ClosestBoundaryPointPRD>();
    const int prim_id = optixGetPrimitiveIndex();
    if (prd.origin_prim_id == prim_id) {
        optixIgnoreIntersection();
        return;
    }

    const owl::vec3f ray_orgin = optixGetWorldRayOrigin();
    const owl::vec3f ray_dir = optixGetWorldRayDirection();
    const float hit_t = optixGetRayTmax();
    const owl::vec3f hit_pos = ray_orgin + hit_t * ray_dir;
    const float dist2 = owl::length2(hit_pos - prd.query_point);

    prd.num_intersections++;
    if (dist2 < prd.dist2) {
        prd.dist2 = dist2;
        prd.position = hit_pos;
        prd.prim_id = prim_id;
    }

    optixIgnoreIntersection();
}

OPTIX_RAYGEN_PROGRAM(windingNumber2DRayGen)() {
    const WindingNumberRayGenData<2> &self = owl::getProgramData<WindingNumberRayGenData<2>>();
    const owl::vec3i idx_3d = optixGetLaunchIndex();

    // round robin scheduling with 32 x indices per chunk.
    const int responsible_device = (idx_3d.x >> 5) % self.device_count;
    if (self.device_index != responsible_device) return;

    const owl::vec3i dim_3d = optixGetLaunchDimensions();
    const int idx = utils::flatten(idx_3d, dim_3d);
    if (idx >= self.num_evaluation_points) return;

    owl::vec2f pos = self.evaluation_point_buf[idx];
    utils::randState_t rand_state = self.random_state_buf[idx];

    if (self.num_primitives > 0)
        self.winding_number_buf[idx] =
            winding_number_estimator<2>(self.acc_structure, pos, rand_state, self.config.num_init_winding_samples);
    else
        self.winding_number_buf[idx] = 0.0f;

    self.winding_number_buf[idx] = owl::clamp(self.winding_number_buf[idx], -1.0f, 1.0f);
    self.random_state_buf[idx] = rand_state;
}

OPTIX_RAYGEN_PROGRAM(windingNumber3DRayGen)() {
    const WindingNumberRayGenData<3> &self = owl::getProgramData<WindingNumberRayGenData<3>>();
    const owl::vec3i idx_3d = optixGetLaunchIndex();

    // round robin scheduling with 32 x indices per chunk.
    const int responsible_device = (idx_3d.x >> 5) % self.device_count;
    if (self.device_index != responsible_device) return;

    const owl::vec3i dim_3d = optixGetLaunchDimensions();
    const int idx = utils::flatten(idx_3d, dim_3d);
    if (idx >= self.num_evaluation_points) return;

    owl::vec3f pos = self.evaluation_point_buf[idx];
    utils::randState_t rand_state = self.random_state_buf[idx];

    if (self.num_primitives > 0)
        self.winding_number_buf[idx] =
            winding_number_estimator<3>(self.acc_structure, pos, rand_state, self.config.num_init_winding_samples);
    else
        self.winding_number_buf[idx] = 0.0f;

    self.winding_number_buf[idx] = owl::clamp(self.winding_number_buf[idx], -1.0f, 1.0f);
    self.random_state_buf[idx] = rand_state;
}
