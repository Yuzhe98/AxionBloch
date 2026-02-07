#include "bloch.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <vector>

namespace py = pybind11;

py::array_t<double> add_3d_arrays_3loops(py::array_t<double> A,
                                  py::array_t<double> B) {
  if (A.ndim() != 3 || B.ndim() != 3)
    throw std::runtime_error("Inputs must be 3D");

  if (A.shape(0) != B.shape(0) || A.shape(1) != B.shape(1) ||
      A.shape(2) != B.shape(2))
    throw std::runtime_error("Shapes do not match");

  // prepare output
  std::vector<py::ssize_t> shape = {A.shape(0), A.shape(1), A.shape(2)};
  // py::array_t<double> C = py::array_t<double>(shape);
  py::array_t<double> C({A.shape(0), A.shape(1), A.shape(2)});

  auto bufA = A.request();
  auto bufB = B.request();
  auto bufC = C.request();

  // native pointers
  const double *pA = static_cast<double *>(bufA.ptr);
  const double *pB = static_cast<double *>(bufB.ptr);
  double *pC = static_cast<double *>(bufC.ptr);

  // call native C++
  _add_3d_arrays_3loops(pA, pB, pC, A.shape(0), A.shape(1), A.shape(2));

  return C;
}

py::array_t<double> add_3d_arrays_parallel(py::array_t<double> A,
                                  py::array_t<double> B) {
  if (A.ndim() != 3 || B.ndim() != 3)
    throw std::runtime_error("Inputs must be 3D");

  if (A.shape(0) != B.shape(0) || A.shape(1) != B.shape(1) ||
      A.shape(2) != B.shape(2))
    throw std::runtime_error("Shapes do not match");

  // prepare output
  std::vector<py::ssize_t> shape = {A.shape(0), A.shape(1), A.shape(2)};
  // py::array_t<double> C = py::array_t<double>(shape);
  py::array_t<double> C({A.shape(0), A.shape(1), A.shape(2)});

  auto bufA = A.request();
  auto bufB = B.request();
  auto bufC = C.request();

  // native pointers
  const double *pA = static_cast<double *>(bufA.ptr);
  const double *pB = static_cast<double *>(bufB.ptr);
  double *pC = static_cast<double *>(bufC.ptr);

  // call native C++
  _add_3d_arrays_parallel(pA, pB, pC, A.shape(0), A.shape(1), A.shape(2));

  return C;
}


py::array_t<double> add_3d_arrays_flattern_parallel(py::array_t<double> A,
                                  py::array_t<double> B) {
  if (A.ndim() != 3 || B.ndim() != 3)
    throw std::runtime_error("Inputs must be 3D");

  if (A.shape(0) != B.shape(0) || A.shape(1) != B.shape(1) ||
      A.shape(2) != B.shape(2))
    throw std::runtime_error("Shapes do not match");

  // prepare output
  std::vector<py::ssize_t> shape = {A.shape(0), A.shape(1), A.shape(2)};
  // py::array_t<double> C = py::array_t<double>(shape);
  py::array_t<double> C({A.shape(0), A.shape(1), A.shape(2)});

  auto bufA = A.request();
  auto bufB = B.request();
  auto bufC = C.request();

  // native pointers
  const double *pA = static_cast<double *>(bufA.ptr);
  const double *pB = static_cast<double *>(bufB.ptr);
  double *pC = static_cast<double *>(bufC.ptr);

  // call native C++
  _add_3d_arrays_flattern_parallel(pA, pB, pC, A.shape(0), A.shape(1), A.shape(2));

  return C;
}

py::array_t<double> add_3d_arrays_flattern_SIMD(py::array_t<double> A,
                                  py::array_t<double> B) {
  if (A.ndim() != 3 || B.ndim() != 3)
    throw std::runtime_error("Inputs must be 3D");

  if (A.shape(0) != B.shape(0) || A.shape(1) != B.shape(1) ||
      A.shape(2) != B.shape(2))
    throw std::runtime_error("Shapes do not match");

  // prepare output
  std::vector<py::ssize_t> shape = {A.shape(0), A.shape(1), A.shape(2)};
  // py::array_t<double> C = py::array_t<double>(shape);
  py::array_t<double> C({A.shape(0), A.shape(1), A.shape(2)});

  auto bufA = A.request();
  auto bufB = B.request();
  auto bufC = C.request();

  // native pointers
  const double *pA = static_cast<double *>(bufA.ptr);
  const double *pB = static_cast<double *>(bufB.ptr);
  double *pC = static_cast<double *>(bufC.ptr);

  // call native C++
  _add_3d_arrays_flattern_SIMD(pA, pB, pC, A.shape(0), A.shape(1), A.shape(2));

  return C;
}

py::array_t<double> burkert_potential_vector(py::array_t<double> r_vals,
 double amp,  double a) {
  if (r_vals.ndim() != 1)
    throw std::runtime_error("r_vals must be 1D");

  // prepare output
  std::vector<py::ssize_t> shape = {r_vals.shape(0)};
  py::array_t<double> pot({r_vals.shape(0)});

  auto bufR = r_vals.request();
  auto bufPot = pot.request();

  // native pointers
  const double *pR = static_cast<double *>(bufR.ptr);
  double *pPot = static_cast<double *>(bufPot.ptr);

  // call native C++
  _burkert_potential_vector(pR, pPot, amp, a, r_vals.shape(0));

  return pot;
}


py::tuple generateTrajectories(py::array_t<double> B_vec,
                                py::array_t<double> dBdt_vec,
                                py::array_t<double> B_vals_T,
                                py::array_t<double> ratios, double gamma,
                                double timeStep, double tSqHalf, double T1,
                                double T2, double RCF_freq_Hz, double Mx0,
                                double My0, double Mz0, double M0eqb) {
  // check input dims
  if (B_vec.ndim() != 3 || dBdt_vec.ndim() != 3)
    throw std::runtime_error("B_vec and dBdt_vec must be 3D");
  if (B_vec.shape(0) != dBdt_vec.shape(0) ||
      B_vec.shape(1) != dBdt_vec.shape(1) ||
      B_vec.shape(2) != dBdt_vec.shape(2))
    throw std::runtime_error("B_vec and dBdt_vec shapes must match");

  if (B_vals_T.ndim() != 1 || ratios.ndim() != 1)
    throw std::runtime_error("B_vals_T and ratios must be 1D");
  if (B_vals_T.shape(0) != ratios.shape(0))
    throw std::runtime_error("B_vals_T and ratios must have same length");

  size_t numFields = B_vec.shape(0);
  size_t numSteps = B_vec.shape(1);
  size_t numSpinPkts = B_vals_T.shape(0);

  std::vector<py::ssize_t> shape_trj = {
      static_cast<py::ssize_t>(numFields),
      static_cast<py::ssize_t>(numSteps + 1), static_cast<py::ssize_t>(3)};

  std::vector<py::ssize_t> shape_time = {static_cast<py::ssize_t>(numFields),
                                         static_cast<py::ssize_t>(numSteps),
                                         static_cast<py::ssize_t>(3)};

  // allocate output arrays
  py::array_t<double> trjry(shape_trj);
  py::array_t<double> dMdt(shape_time);
  py::array_t<double> McrossB(shape_time);
  py::array_t<double> d2Mdt2(shape_time);

  // get raw pointers
  auto bufB = B_vec.request();
  auto bufdB = dBdt_vec.request();
  auto bufBvals = B_vals_T.request();
  auto bufratios = ratios.request();
  auto bufTrj = trjry.request();
  auto bufdMdt = dMdt.request();
  auto bufMcrossB = McrossB.request();
  auto bufd2Mdt2 = d2Mdt2.request();

  const double *pB = static_cast<double *>(bufB.ptr);
  const double *pdB = static_cast<double *>(bufdB.ptr);
  const double *pBvals = static_cast<double *>(bufBvals.ptr);
  const double *pratios = static_cast<double *>(bufratios.ptr);

  double *ptrj = static_cast<double *>(bufTrj.ptr);
  double *pdM = static_cast<double *>(bufdMdt.ptr);
  double *pCross = static_cast<double *>(bufMcrossB.ptr);
  double *pd2M = static_cast<double *>(bufd2Mdt2.ptr);

  // call C function
  _generateTrajectories(
      // inputs
      static_cast<int>(numFields), static_cast<int>(numSteps),
      static_cast<int>(numSpinPkts), // int
      pB, // shape (numFields, numSteps, 3)
      pdB,                               // shape (numFields, numSteps, 3)
      pBvals,                            // shape (numSpinPkts)
      pratios,                           // shape (numSpinPkts)
      gamma, timeStep, tSqHalf, T1, T2, RCF_freq_Hz, 
      Mx0, My0, Mz0, M0eqb, // initial magnetization
      // ouputs
      ptrj, pdM, pCross, pd2M);

  return py::make_tuple(trjry, dMdt, McrossB, d2Mdt2);
}

PYBIND11_MODULE(blochSimulation_c, m) {
  m.doc() = "Native C++ for nuclear magnetic resonance simulation based on Bloch equations. ";
  m.def("add_3d_arrays_3loops", &add_3d_arrays_3loops, "Add two X×Y×Z arrays");
  m.def("add_3d_arrays_parallel", &add_3d_arrays_parallel, "Add two X×Y×Z arrays");
  m.def("add_3d_arrays_flattern_parallel", &add_3d_arrays_flattern_parallel, "Add two X×Y×Z arrays");
  m.def("add_3d_arrays_flattern_SIMD", &add_3d_arrays_flattern_SIMD, "Add two X×Y×Z arrays");
  // Scalar function
  // m.def("burkert_potential", &burkert_potential, "Compute Burkert potential at a single r");
  // Vectorized function
  m.def("burkert_potential_vector", &burkert_potential_vector, "Compute Burkert potential for a list of r values");
  
  // new Bloch trajectory function
  m.def("generateTrajectories", &generateTrajectories,
        "Compute magnetization trajectories (trjry, dMdt, McrossB, d2Mdt2)");
}
