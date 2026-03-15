#include "bloch.h"
#include <immintrin.h>
#include <omp.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void _add_3d_arrays_3loops(const double *A, const double *B, double *C, std::size_t X,
                           std::size_t Y, std::size_t Z) {
    std::size_t N = X * Y * Z;

    for (int i = 0; i < X; i++) {
        for (int j = 0; j < Y; j++) {
            for (int k = 0; k < Z; k++) {
                std::size_t idx = i * (Y * Z) + j * Z + k;
                C[idx] = A[idx] + B[idx];
            }
        }
    }
}

void _add_3d_arrays_parallel(const double *A, const double *B, double *C, std::size_t X,
                             std::size_t Y, std::size_t Z) {
    std::size_t N = X * Y * Z;

#pragma omp parallel for schedule(static)
    for (int i = 0; i < X; i++) {
        for (int j = 0; j < Y; j++) {
            for (int k = 0; k < Z; k++) {
                std::size_t idx = i * (Y * Z) + j * Z + k;
                C[idx] = A[idx] + B[idx];
            }
        }
    }
}

void _add_3d_arrays_flattern_parallel(const double *A, const double *B, double *C, std::size_t X,
                                      std::size_t Y, std::size_t Z) {
    std::size_t N = X * Y * Z;

#pragma omp parallel for schedule(static)
    for (long idx = 0; idx < N; ++idx) {
        C[idx] = A[idx] + B[idx];
    }
}

void _add_3d_arrays_flattern_SIMD(const double *A, const double *B, double *C, std::size_t X,
                                  std::size_t Y, std::size_t Z) {
    std::size_t N = X * Y * Z;
    std::size_t i = 0;

#ifdef __AVX2__
    for (; i + 3 < N; i += 4) {
        __m256d a = _mm256_loadu_pd(A + i);
        __m256d b = _mm256_loadu_pd(B + i);
        __m256d c = _mm256_add_pd(a, b);
        _mm256_storeu_pd(C + i, c);
    }
#endif

    // handle remaining elements
    for (; i < N; ++i)
        C[i] = A[i] + B[i];
}

// Scalar function
double _burkert_potential(double r, double amp, double a) {
    return amp / ((1 + r / a) * (1 + (r / a) * (r / a)));
}

// Vectorized function
// std::vector<double> _burkert_potential_vector(const double *r_vals,
//                                              const double amp, const double a) {
//     std::vector<double> out(r_vals.size());
//     for (size_t i = 0; i < r_vals.size(); ++i) {
//         out[i] = _burkert_potential(r_vals[i], amp, a);
//     }
//     return out;
// }

void _burkert_potential_vector(const double *r_vals, double *pot, const double amp, const double a,
                               std::size_t N) {
    std::size_t i = 0;
    // handle remaining elements
    for (; i < N; ++i)
        pot[i] = _burkert_potential(r_vals[i], amp, a);
}

void _generateTrajectories_TE(
    // input
    const int numFields, const int numTimeSteps, const int numSpinPkts,
    const double *B_vec,    // shape (numFields, numTimeSteps, 3) -> 1D
    const double *dBdt_vec, // shape (numFields, numTimeSteps, 3) -> 1D
    const double *B_vals_T, // shape (numSpinPkts)
    const double *ratios,   // shape (numSpinPkts)
    const double gamma, const double timeStep, const double T1, const double T2,
    const double RCF_freq_Hz, const double Mx0, const double My0, const double Mz0,
    const double M0eqb,
    // output
    double *trjry, // shape (numFields, numTimeSteps+1, 3) -> 1D
    double *dMdt,  // shape (numFields, numTimeSteps, 3) -> 1D
    double *d2Mdt2 // shape (numFields, numTimeSteps, 3) -> 1D
) {
    double gamma2pi = gamma / (2 * M_PI);
    double tSqHalf = 0.5 * timeStep * timeStep; // half timestep squared
// parallelize over fields
#pragma omp parallel for schedule(static)
    for (int f = 0; f < numFields; f++) {

        // Allocate spin packet arrays
        double *Mx = (double *)malloc(numSpinPkts * sizeof(double));
        double *My = (double *)malloc(numSpinPkts * sizeof(double));
        double *Mz = (double *)malloc(numSpinPkts * sizeof(double));
        double *M0eq = (double *)malloc(numSpinPkts * sizeof(double));
        double *B0z_rot_amp = (double *)malloc(numSpinPkts * sizeof(double));

        // initialize spin packets
        for (int k = 0; k < numSpinPkts; k++) {
            Mx[k] = ratios[k] * Mx0;
            My[k] = ratios[k] * My0;
            Mz[k] = ratios[k] * Mz0;
            M0eq[k] = ratios[k] * M0eqb;
            B0z_rot_amp[k] = B_vals_T[k] - RCF_freq_Hz / gamma2pi;
        }

        // write initial values in trjry
        // in the array (f-th, numTimeSteps + 1, 3)
        int idx0 = (f * (numTimeSteps + 1) + 0) * 3;
        trjry[idx0 + 0] = Mx0;
        trjry[idx0 + 1] = My0;
        trjry[idx0 + 2] = Mz0;

        // time loop
        for (int t = 0; t < numTimeSteps; t++) {

            // use B field in the array (f-th, t, :)
            int idxB = (f * numTimeSteps + t) * 3;
            double Bx = B_vec[idxB + 0];
            double By = B_vec[idxB + 1];
            double Bz_raw = B_vec[idxB + 2];

            double dBxdt = dBdt_vec[idxB + 0];
            double dBydt = dBdt_vec[idxB + 1];
            double dBzdt = dBdt_vec[idxB + 2];

            double sumMx = 0.0, sumMy = 0.0, sumMz = 0.0;

            // vectorized loop over spin packets
            for (int k = 0; k < numSpinPkts; k++) {

                double Bz = Bz_raw + B0z_rot_amp[k];

                // first derivative
                double dMxdt = gamma * (My[k] * Bz - Mz[k] * By) - Mx[k] / T2;
                double dMydt = gamma * (Mz[k] * Bx - Mx[k] * Bz) - My[k] / T2;
                double dMzdt = gamma * (Mx[k] * By - My[k] * Bx) - (Mz[k] - M0eq[k]) / T1;

                // second derivative
                double d2Mxdt2 =
                    gamma * (dMydt * Bz + My[k] * dBzdt - dMzdt * By - Mz[k] * dBydt) - dMxdt / T2;

                double d2Mydt2 =
                    gamma * (dMzdt * Bx + Mz[k] * dBxdt - dMxdt * Bz - Mx[k] * dBzdt) - dMydt / T2;

                double d2Mzdt2 =
                    gamma * (dMxdt * By + Mx[k] * dBydt - dMydt * Bx - My[k] * dBxdt) - dMzdt / T1;

                // update
                Mx[k] += dMxdt * timeStep + tSqHalf * d2Mxdt2;
                My[k] += dMydt * timeStep + tSqHalf * d2Mydt2;
                Mz[k] += dMzdt * timeStep + tSqHalf * d2Mzdt2;

                // accumulate
                sumMx += Mx[k];
                sumMy += My[k];
                sumMz += Mz[k];
            }

            // store trajectory for field f
            // write in the array (f-th, t+1, :)
            int idxOut = (f * (numTimeSteps + 1) + (t + 1)) * 3;
            trjry[idxOut + 0] = sumMx;
            trjry[idxOut + 1] = sumMy;
            trjry[idxOut + 2] = sumMz;
        }

        free(Mx);
        free(My);
        free(Mz);
        free(M0eq);
        free(B0z_rot_amp);
    }
}

/**
 * using Runge–Kutta 4 (RK4) method for solving the Bloch equations
 * 
 */
void _generateTrajectories(
    // input
    const int numFields, const int numTimeSteps, const int numSpinPkts,
    const double *B_vec,    // shape (numFields, numTimeSteps, 3) -> 1D
    const double *dBdt_vec, // shape (numFields, numTimeSteps, 3) -> 1D
    const double *B_vals_T, // shape (numSpinPkts)
    const double *ratios,   // shape (numSpinPkts)
    const double gamma, const double timeStep, const double T1,
    const double T2, const double RCF_freq_Hz, const double Mx0, const double My0, const double Mz0,
    const double M0eqb,
    // output
    double *trjry,   // shape (numFields, numTimeSteps+1, 3) -> 1D
    double *dMdt,    // shape (numFields, numTimeSteps, 3) -> 1D
    double *d2Mdt2   // shape (numFields, numTimeSteps, 3) -> 1D
) {
    double dt = timeStep;
    double dt_2 = 0.5 * dt; // half timestep
    double dt_6 = dt / 6.0; // for final RK4 update

    const int FT3 = numFields * numTimeSteps * 3;

    std::vector<double> Omega_vec(FT3);
    std::vector<double> dOmegadt_vec(FT3);
    for (int i = 0; i < FT3; i++) {
        Omega_vec[i] = gamma * B_vec[i];
        dOmegadt_vec[i] = gamma * dBdt_vec[i];
    }

    std::vector<double> inhomoOmega_vec(numSpinPkts);
    for (int i = 0; i < numSpinPkts; i++) {
        inhomoOmega_vec[i] = gamma * B_vals_T[i];
    }

// parallelize over fields
#pragma omp parallel for schedule(static)
    for (int f = 0; f < numFields; f++) {

        // Allocate spin packet arrays
        double *Mx = (double *)malloc(numSpinPkts * sizeof(double));
        double *My = (double *)malloc(numSpinPkts * sizeof(double));
        double *Mz = (double *)malloc(numSpinPkts * sizeof(double));
        double *M0eq = (double *)malloc(numSpinPkts * sizeof(double));
        double *inhomoOz_vec = (double *)malloc(numSpinPkts * sizeof(double));

        // initialize spin packets
        for (int k = 0; k < numSpinPkts; k++) {
            Mx[k] = ratios[k] * Mx0;
            My[k] = ratios[k] * My0;
            Mz[k] = ratios[k] * Mz0;
            M0eq[k] = ratios[k] * M0eqb;
            inhomoOz_vec[k] = inhomoOmega_vec[k] - RCF_freq_Hz * 2 * M_PI;
        }

        // write initial values in trjry
        // in the array (f-th, numTimeSteps + 1, 3)
        int idx0 = (f * (numTimeSteps + 1) + 0) * 3;
        trjry[idx0 + 0] = Mx0;
        trjry[idx0 + 1] = My0;
        trjry[idx0 + 2] = Mz0;

        // time loop
        for (int t = 0; t < numTimeSteps; t++) {

            // use B field in the array (f-th, t, :)
            int idxB = (f * numTimeSteps + t) * 3;
            double Ox = Omega_vec[idxB + 0];
            double Oy = Omega_vec[idxB + 1];
            double Oz_raw = Omega_vec[idxB + 2];

            double dOxdt = dOmegadt_vec[idxB + 0];
            double dOydt = dOmegadt_vec[idxB + 1];
            double dOzdt = dOmegadt_vec[idxB + 2];

            double sumMx = 0.0, sumMy = 0.0, sumMz = 0.0;

            // vectorized loop over spin packets
            for (int k = 0; k < numSpinPkts; k++) {

                double Oz = Oz_raw + inhomoOz_vec[k];

                // RK4 integration for spin packet k

                // Step 1: compute k1
                double k1x = (My[k] * Oz - Mz[k] * Oy) - Mx[k] / T2;
                double k1y = (Mz[k] * Ox - Mx[k] * Oz) - My[k] / T2;
                double k1z = (Mx[k] * Oy - My[k] * Ox) - (Mz[k] - M0eq[k]) / T1;

                // Step 2: compute k2
                double Mx_tmp = Mx[k] + dt_2 * k1x;
                double My_tmp = My[k] + dt_2 * k1y;
                double Mz_tmp = Mz[k] + dt_2 * k1z;

                double k2x = (My_tmp * Oz - Mz_tmp * Oy) - Mx_tmp / T2;
                double k2y = (Mz_tmp * Ox - Mx_tmp * Oz) - My_tmp / T2;
                double k2z = (Mx_tmp * Oy - My_tmp * Ox) - (Mz_tmp - M0eq[k]) / T1;

                // Step 3: compute k3
                Mx_tmp = Mx[k] + dt_2 * k2x;
                My_tmp = My[k] + dt_2 * k2y;
                Mz_tmp = Mz[k] + dt_2 * k2z;

                double k3x = (My_tmp * Oz - Mz_tmp * Oy) - Mx_tmp / T2;
                double k3y = (Mz_tmp * Ox - Mx_tmp * Oz) - My_tmp / T2;
                double k3z = (Mx_tmp * Oy - My_tmp * Ox) - (Mz_tmp - M0eq[k]) / T1;

                // Step 4: compute k4
                Mx_tmp = Mx[k] + timeStep * k3x;
                My_tmp = My[k] + timeStep * k3y;
                Mz_tmp = Mz[k] + timeStep * k3z;

                double k4x = (My_tmp * Oz - Mz_tmp * Oy) - Mx_tmp / T2;
                double k4y = (Mz_tmp * Ox - Mx_tmp * Oz) - My_tmp / T2;
                double k4z = (Mx_tmp * Oy - My_tmp * Ox) - (Mz_tmp - M0eq[k]) / T1;

                // Step 5: update M
                Mx[k] += dt_6 * (k1x + 2 * k2x + 2 * k3x + k4x);
                My[k] += dt_6 * (k1y + 2 * k2y + 2 * k3y + k4y);
                Mz[k] += dt_6 * (k1z + 2 * k2z + 2 * k3z + k4z);

                // accumulate total magnetization
                sumMx += Mx[k];
                sumMy += My[k];
                sumMz += Mz[k];
            }

            // store trajectory for field f
            // write in the array (f-th, t+1, :)
            int idxOut = (f * (numTimeSteps + 1) + (t + 1)) * 3;
            trjry[idxOut + 0] = sumMx;
            trjry[idxOut + 1] = sumMy;
            trjry[idxOut + 2] = sumMz;
        }

        free(Mx);
        free(My);
        free(Mz);
        free(M0eq);
        free(inhomoOz_vec);
    }
}
