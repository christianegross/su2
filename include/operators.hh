// operators.hh
/**
 * @brief different operators on the lattice which can be measured for the use in calculation of the glueball mass
 * In contrast to the gauge_energy or Wilson-loop operators, these operators should only ever look at one timeslice
 */

#pragma once

#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

#include "gradient_flow.hh"
#include "parameters.hh"
#include "propagator.hpp"
#include "wilsonloop.hh"
#include "gaugeconfig.hh"
#include "vectorfunctions.hh"

using Complex = std::complex<double>;

namespace operators {
    /**
     * @brief computes the sum of all plaquettes in the mu-nu-plane in the time slice t
     * **/
  template <class Group>
  double ReP(const gaugeconfig<Group> &U,
                        const size_t &t,
                        const size_t &mu,
                        const size_t &nu) {
    double res = 0.;
    typedef typename accum_type<Group>::type accum;
    
    #pragma omp parallel for reduction(+ : res)
    for (size_t x = 0; x < U.getLx(); x++) {
      for (size_t y = 0; y < U.getLy(); y++) {
        for (size_t z = 0; z < U.getLz(); z++) {
          std::vector<size_t> vecx = {t, x, y, z};
          std::vector<size_t> vecxplusmu = vecx;
          std::vector<size_t> vecxplusnu = vecx;
          vecxplusmu[mu] += 1;
          vecxplusnu[nu] += 1;
          res += retrace(U(vecx, mu) * U(vecxplusmu, nu) * U(vecxplusnu, mu).dagger() *
                          U(vecx, nu).dagger());
          vecxplusmu[mu] -= 1;
          vecxplusnu[nu] -= 1;
        }
      }
    }
    return res;
  }
  
  /**
   * @brief This multiplies all the link matrices along a given path and stores the results in the group accum type
   * The path starts at x, and is described by the vectors lengths, directions, and sign
   * @param U configuration upon which the operator is measured
   * @param x vector of the starting point of the operator
   * @param lengths, directions, sign vectors describing the path of the operator
   * @note The vectors must have the same length. For each element, the path goes lengths[i] steps into direction[i],
   * with sign[i] determining if the path is traced forwards or backwards.
   * It makes sense to have all elements of direction in (0, U.getndims-1), 
   * but if this is not the case, the internal function getindex will ensure that directions[i] is interpreted as directions[i]%U.getndims
   * @return accum type of the group of the ordered product of all matrices in the operator, so Complex for U(1) and SU(2) for SU(2). 
   * This makes it easier to take either the real or imaginary part, with or without the trace
   * Maybe it makes more sense to return the trace in a complex number? Are there any cases where the trace is not used in an operator?
   * @note example: P_mu,nu(x): lengths={1,1,1,1}, directions={mu, nu, mu, nu}, sign={true, true, false, false}
   * example: W(t=1, x=4): lengths={1,4,1,4}, directions={0,1,0,1}, sign={true, true, false, false}
   * example: chair xz-yz: lengths={1,1,1,1,1,1}, directions={1,2,3,2,1,3}, sign={t,t,t,f,f,f}
   * example: chair xy-yz: lengths={1,1,1,1,1,1}, directions={1,3,2,3,2,1}, sign={t,t,t,f,f,f}
   * **/
  template <class Group>
  typename accum_type<Group>::type arbitrary_operator(const gaugeconfig<Group>&U, 
                                             const std::vector<size_t> &x,
                                             const std::vector<size_t> &lengths,
                                             const std::vector<size_t> &directions,
                                             const std::vector<bool> &sign){
    typedef typename accum_type<Group>::type accum;
    if( (lengths.size()!=directions.size()) || (lengths.size()!=sign.size()) ){
      std::cerr << "the lengths of the descriptors of the loop are not equal, no loop can be calculated!" << std::endl;
      abort();
    }
    std::vector<size_t> xrun=x;
    accum L(1., 0.);
    for(size_t i=0; i<lengths.size(); i++){
      if(sign[i]){
        for(size_t j=0; j<lengths[i]; j++){
          L *= U(xrun, directions[i]);
          xrun[directions[i]]++;
        }
      }
      if(!sign[i]){
        for(size_t j=0; j<lengths[i]; j++){
          xrun[directions[i]]--;
          L *= U(xrun, directions[i]).dagger();
        }
      }
    }
    if(xrun != x){
      std::cerr << "The loop was not closed!" << std::endl;
      //~ abort();
    }
    return L;
  }
  
  /** 
   * @brief returns the sum of the real trace of the arbitrary operator 
   * given by the vectors lengths, directions and sign in the timeslice with index t
   * see documentation of arbitrary_operator
   * **/
  template <class Group>
  double measure_re_arbitrary_loop_one_timeslice(const gaugeconfig<Group>&U,
                                          const size_t t,
                                          const std::vector<size_t> &lengths,
                                          const std::vector<size_t> &directions,
                                          const std::vector<bool> &sign){
    double res = 0.;
    typedef typename accum_type<Group>::type accum;
    
    #pragma omp parallel for reduction(+ : res)
    for (size_t x = 0; x < U.getLx(); x++) {
      for (size_t y = 0; y < U.getLy(); y++) {
        for (size_t z = 0; z < U.getLz(); z++) {
          std::vector<size_t> vecx = {t, x, y, z};
          res += retrace(arbitrary_operator(U, vecx, lengths, directions, sign));
        }
      }
    }
    return res;
  }
    
  /** 
   * @brief returns the sum of the real trace of the arbitrary operator 
   * given by the vectors lengths, directions and sign calculated in the entire lattice
   * see documentation of arbitrary_operator
   * **/
  template <class Group>
  double measure_re_arbitrary_loop_lattice(const gaugeconfig<Group>&U,
                                    const std::vector<size_t> &lengths,
                                    const std::vector<size_t> &directions,
                                    const std::vector<bool> &sign){
    double res = 0.;
    typedef typename accum_type<Group>::type accum;
    
    #pragma omp parallel for reduction(+ : res)
    for (size_t t = 0; t < U.getLt(); t++) {
      for (size_t x = 0; x < U.getLx(); x++) {
        for (size_t y = 0; y < U.getLy(); y++) {
          for (size_t z = 0; z < U.getLz(); z++) {
            std::vector<size_t> vecx = {t, x, y, z};
            res += retrace(arbitrary_operator(U, vecx, lengths, directions, sign));
          }
        }
      }
    }
    return res;
  }
  
} //namespace operators

//tests done with
    //~ double loop = operators::measure_re_arbitrary_loop_lattice(U, {1,1,1,1}, {1,2,1,2}, {true, true, false, false});
    //~ loop += operators::measure_re_arbitrary_loop_lattice(U, {1,1,1,1}, {1,3,1,3}, {true, true, false, false});
    //loop += operators::measure_re_arbitrary_loop_lattice(U, {1,1,1,1}, {2,3,2,3}, {true, true, false, false});
    //std::cout << loop/U.getVolume()*2./U.getndims()/(U.getndims()-1) << " ";
    //~ std::cout << loop/U.getVolume()/2 << " ";
    //~ loop = operators::measure_re_arbitrary_loop_lattice(U, {2,1,2,1}, {0,1,0,1}, {true, true, false, false});
    //loop += operators::measure_re_arbitrary_loop_lattice(U, {1,1,1,1}, {0,2,0,2}, {true, true, false, false});
    //loop += operators::measure_re_arbitrary_loop_lattice(U, {1,1,1,1}, {0,3,0,3}, {true, true, false, false});
    //std::cout << loop/U.getVolume()*2./U.getndims()/(U.getndims()-1) << std::endl;
    //~ std::cout << loop/U.getVolume() << std::endl;
// in measure-u1 agree with results of Wilson-Loops
