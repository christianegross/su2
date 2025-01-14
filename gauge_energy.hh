#pragma once
#include"gaugeconfig.hh"
#ifdef _USE_OMP_
#  include<omp.h>
#endif

// this is the Wilson plaquette gauge energy
// \sum_mu \sum_nu<mu tr(P_{mu nu})
//
// checked for gauge invariance

template<class T> double gauge_energy(gaugeconfig<T> &U, bool spacial=false) {

  double res = 0.;
#ifdef _USE_OMP_
  int threads = omp_get_max_threads();
  static double * omp_acc = new double[threads];
#pragma omp parallel
  {
    int thread_num = omp_get_thread_num();
#endif
    double tmp = 0.;
    
size_t startmu=0;
if(spacial){startmu=1;};
 
#pragma omp for
    for(size_t x0 = 0; x0 < U.getLt(); x0++) {
      for(size_t x1 = 0; x1 < U.getLx(); x1++) {
        for(size_t x2 = 0; x2 < U.getLy(); x2++) {
          for(size_t x3 = 0; x3 < U.getLz(); x3++) {
            std::vector<size_t> x = {x0, x1, x2, x3};
            std::vector<size_t> xplusmu = x;
            std::vector<size_t> xplusnu = x;
            for(size_t mu = startmu; mu < U.getndims()-1; mu++) {
              for(size_t nu = mu+1; nu < U.getndims(); nu++) {
                xplusmu[mu] += 1;
                xplusnu[nu] += 1;
                tmp += retrace(U(x, mu) * U(xplusmu, nu) *
                               U(xplusnu, mu).dagger()*U(x, nu).dagger());
                xplusmu[mu] -= 1;
                xplusnu[nu] -= 1;
              }
            }
          }
        }
      }
    }
#ifdef _USE_OMP_
    omp_acc[thread_num] = tmp;
    res = 0.;
  }
  for(size_t i = 0; i < threads; i++) {
    res += omp_acc[i];
  }
#else
  res = tmp;
#endif
  return res;
}
