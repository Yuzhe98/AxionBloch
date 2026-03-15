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

/**
 * @brief Generate trajectories for the Bloch equations
 *
 * @param numFields
 * @param numTimeSteps
 * @param numSpinPkts
 * @param B_vec vector of B fields, shape (numFields, numTimeSteps, 3)
 *      B_vec[(f * numTimeSteps + t) * 3 + 0 or 1 or 2] is Bx or By or Bz for field f at time step t
 * @param dBdt_vec
 * @param B_vals_T
 * @param ratios
 * @param gamma
 * @param timeStep
 * @param tSqHalf
 * @param T1
 * @param T2
 * @param RCF_freq_Hz
 * @param Mx0
 * @param My0
 * @param Mz0
 * @param M0eqb
 * @param trjry
 * @param dMdt
 * @param McrossB
 * @param d2Mdt2
 */
void _generateTrajectories(
    // input
    const int numFields, const int numTimeSteps, const int numSpinPkts,
    const double *B_vec,    // shape (numFields, numTimeSteps, 3)
    const double *dBdt_vec, // shape (numFields, numTimeSteps, 3)
    const double *B_vals_T, // shape (numSpinPkts)
    const double *ratios,   // shape (numSpinPkts)
    const double gamma, const double timeStep, const double tSqHalf, const double T1,
    const double T2, const double RCF_freq_Hz, const double Mx0, const double My0, const double Mz0,
    const double M0eqb,
    // output
    double *trjry,   // shape (numFields, numTimeSteps+1, 3)
    double *dMdt,    // shape (numFields, numTimeSteps, 3)
    double *McrossB, // shape (numFields, numTimeSteps, 3)
    double *d2Mdt2   // shape (numFields, numTimeSteps, 3)
) {
    double gamma2pi = gamma / (2 * M_PI);

// parallelize over fields
#pragma omp parallel for schedule(static)
    for (int f = 0; f < numFields; f++) {

        // Allocate spin packet arrays
        double *Mx = (double *)malloc(numSpinPkts * sizeof(double));
        double *My = (double *)malloc(numSpinPkts * sizeof(double));
        double *Mz = (double *)malloc(numSpinPkts * sizeof(double));
        double *M0eq = (double *)malloc(numSpinPkts * sizeof(double));
        double *B0z_rot_amp = (double *)malloc(numSpinPkts * sizeof(double));

        double *dMxdt = (double *)malloc(numSpinPkts * sizeof(double));
        double *dMydt = (double *)malloc(numSpinPkts * sizeof(double));
        double *dMzdt = (double *)malloc(numSpinPkts * sizeof(double));

        double *d2Mxdt2 = (double *)malloc(numSpinPkts * sizeof(double));
        double *d2Mydt2 = (double *)malloc(numSpinPkts * sizeof(double));
        double *d2Mzdt2 = (double *)malloc(numSpinPkts * sizeof(double));

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

            // 0th is current Mx, My, Mz (state is already stored)

            // 1st derivative pass
            for (int k = 0; k < numSpinPkts; k++) {
                double Bz = Bz_raw + B0z_rot_amp[k];

                dMxdt[k] = gamma * (My[k] * Bz - Mz[k] * By) - Mx[k] / T2;
                dMydt[k] = gamma * (Mz[k] * Bx - Mx[k] * Bz) - My[k] / T2;
                dMzdt[k] = gamma * (Mx[k] * By - My[k] * Bx) - (Mz[k] - M0eq[k]) / T1;
            }

            // 2nd derivative pass
            for (int k = 0; k < numSpinPkts; k++) {
                double Bz = Bz_raw + B0z_rot_amp[k];

                d2Mxdt2[k] =
                    gamma * (dMydt[k] * Bz + My[k] * dBzdt - dMzdt[k] * By - Mz[k] * dBydt) -
                    dMxdt[k] / T2;
                d2Mydt2[k] =
                    gamma * (dMzdt[k] * Bx + Mz[k] * dBxdt - dMxdt[k] * Bz - Mx[k] * dBzdt) -
                    dMydt[k] / T2;
                d2Mzdt2[k] =
                    gamma * (dMxdt[k] * By + Mx[k] * dBydt - dMydt[k] * Bx - My[k] * dBxdt) -
                    dMzdt[k] / T1;
            }

            // update M and accumulate
            for (int k = 0; k < numSpinPkts; k++) {
                Mx[k] += dMxdt[k] * timeStep + tSqHalf * d2Mxdt2[k];
                My[k] += dMydt[k] * timeStep + tSqHalf * d2Mydt2[k];
                Mz[k] += dMzdt[k] * timeStep + tSqHalf * d2Mzdt2[k];

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
        free(dMxdt);
        free(dMydt);
        free(dMzdt);
        free(d2Mxdt2);
        free(d2Mydt2);
        free(d2Mzdt2);
    }
}
