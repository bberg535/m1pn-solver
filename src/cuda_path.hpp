#pragma once
#include "angular.hpp"
#include "spatial.hpp"
namespace m1pn {
// Explicit runtime switch. Unset leaves the CPU path. M1PN_CUDA=1 requests
// CUDA unless the study CLI overrides it with --cuda off.
void cuda_set_requested(bool on);
bool cuda_requested();
// Upload A, C and the lumped mass operator once. Throws if CUDA was requested
// but this binary or the device cannot run the kernels. Never falls back.
void cuda_prepare(const Angular &pn, const Spatial &fe);
bool cuda_ready();
void cuda_fill_angular_products(const Angular &pn, const Field &state, std::array<Field, 3> &flux);
void cuda_add_directional_scatter(Field &destination, const std::array<Sparse, 3> &matrices,
                                  const std::array<Field, 3> &flux);
void cuda_mass_correction(const Spatial &fe, const Field &rate, Field &correction);
} // namespace m1pn
