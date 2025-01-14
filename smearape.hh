#pragma once

#include"su2.hh"
#include"u1.hh"
#include"random_element.hh"
#include"gaugeconfig.hh"
#include"get_staples.hh"
#include<random>
#include<vector>
#include<cmath>
#include<fstream>
#include<complex>
#include<iostream>
#include<cassert>
#ifdef _USE_OMP_
#  include<omp.h>
#endif


//prescription for smearing taken from https://arxiv.org/abs/hep-lat/0209159
//not tested for SU2, for U1 lattice stays the same when smearing with alpha=1
//~ template<class Group> void smearlatticeape(gaugeconfig<Group> &U, double alpha, const double xi=1.0, bool anisotropic=false, bool spacial=false){
template<class Group> void smearlatticeape(gaugeconfig<Group> &U, double alpha, bool spacial=false){
  gaugeconfig<Group> Uold = U;
  typedef typename accum_type<Group>::type accum;
  size_t startmu=0;
  if(spacial){
      startmu=1;
  }
  #ifdef _USE_OMP_
  #pragma omp parallel for
  #endif
  for(size_t x0 = 0; x0 < U.getLt(); x0++) {
    for(size_t x1 = 0; x1 < U.getLx(); x1++) {
      for(size_t x2 = 0; x2 < U.getLy(); x2++) {
        for(size_t x3 = 0; x3 < U.getLz(); x3++) {
          std::vector<size_t> x = {x0, x1, x2, x3};  
          for(size_t mu = startmu; mu < U.getndims(); mu++) {
            accum K;
            //~ get_staples(K, Uold, x, mu, xi, anisotropic, spacial);
            get_staples(K, Uold, x, mu, 1.0, false, spacial);
            Group Uprime(Uold(x, mu)*alpha+K*(1-alpha)/2.0);
            U(x, mu) = Uprime;
            U(x, mu).restoreSU();
          }
        }
      }
    }
  }      
}

