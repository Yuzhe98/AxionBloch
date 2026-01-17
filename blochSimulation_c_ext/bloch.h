#pragma once

#include <cstddef>
#include <math.h>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

/* Ensure M_PI is defined */
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/***************************************
 * 3D array addition utilities
 ***************************************/
void _add_3d_arrays_3loops(const double *A, const double *B, double *C,
                           std::size_t X, std::size_t Y, std::size_t Z);

void _add_3d_arrays_parallel(const double *A, const double *B, double *C,
                             std::size_t X, std::size_t Y, std::size_t Z);

void _add_3d_arrays_flattern_parallel(const double *A, const double *B,
                                      double *C, std::size_t X, std::size_t Y,
                                      std::size_t Z);

void _add_3d_arrays_flattern_SIMD(const double *A, const double *B, double *C,
                                  std::size_t X, std::size_t Y, std::size_t Z);


// Scalar Burkert potential
double _burkert_potential(double r, double amp, double a);

// Vectorized Burkert potential
void _burkert_potential_vector(const double *r_vals, double *pot, const double amp, const double a, std::size_t N);


/***************************************
 * Magnetization trajectory generator
 ***************************************/

/**
 * _generateTrajectories
 *
 * Computes magnetization trajectories for multiple independent fields,
 * vectorized over spin packets, and parallelized over fields using OpenMP.
 */
void _generateTrajectories(
    // input
    const int numFields, const int numTimeSteps, const int numSpinPkts,
    const double *B_vec,    // shape (numFields, numTimeSteps, 3)
    const double *dBdt_vec, // shape (numFields, numTimeSteps, 3)
    const double *B_vals_T, // shape (numSpinPkts)
    const double *ratios,   // shape (numSpinPkts)
    const double gamma, const double timeStep, const double tSqHalf,
    const double T1, const double T2, const double RCF_freq_Hz,
    const double Mx0, const double My0, const double Mz0, const double M0eqb,

    // output
    double *trjry,   // shape (numFields, numTimeSteps+1, 3)
    double *dMdt,    // shape (numFields, numTimeSteps, 3)
    double *McrossB, // shape (numFields, numTimeSteps, 3)
    double *d2Mdt2   // shape (numFields, numTimeSteps, 3)
);

#ifdef __cplusplus
}
#endif
