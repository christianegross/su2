// omeasurements.hpp
/**
 * @brief routines for online/offline measurements of various observables.
 * This file defines functions which handle the measure and printing of observables over a
 * given gauge configuration.
 */

#pragma once

#include <iomanip>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>

#include "gradient_flow.hh"
#include "parameters.hh"
#include "propagator.hpp"
#include "wilsonloop.hh"
#include "operators.hh"

namespace omeasurements {

  /**
   * @brief compute and print the wilson loop of a given configuration
   *
   * @tparam Group
   * @param U gauge config
   * @param i configurationn index
   * @param conf_dir
   */
  template <class Group>
  void meas_wilson_loop(const gaugeconfig<Group> &U,
                        const size_t &i,
                        const std::string &conf_dir) {
    std::ostringstream os;
    os << conf_dir + "/wilsonloop.";
    auto prevw = os.width(6);
    auto prevf = os.fill('0');
    os << i;
    os.width(prevw);
    os.fill(prevf);
    os << ".dat" << std::ends;
    compute_all_loops(U, os.str());

    return;
  }

  /**
   * @brief compute and print the gradient flow of a given configuration
   *
   * @tparam Group
   * @param U gauge config
   * @param i configurationn index
   * @param conf_dir
   */
  template <class Group>
  void meas_gradient_flow(const gaugeconfig<Group> &U,
                          const size_t &i,
                          const std::string &conf_dir,
                          const double &tmax) {
    std::ostringstream os;
    os << conf_dir + "/gradient_flow.";
    auto prevw = os.width(6);
    auto prevf = os.fill('0');
    os << i;
    os.width(prevw);
    os.fill(prevf);
    gradient_flow(U, os.str(), tmax);

    return;
  }

  /**
   * @brief measure and print the (staggered) pion correlator
   * It is assumed that sparams (specific parameters) contain the following attributes:
   * -
   * @tparam Group
   * @tparam sparams struct containing info on computation and output
   * @param U gauge configuration
   * @param i trajectory index
   * @param S
   */
  template <class Group, class sparams>
  void
  meas_pion_correlator(const gaugeconfig<Group> &U, const size_t &i, const double& m, const sparams &S) {
    std::ostringstream oss;
    oss << S.conf_dir + "/C_pion.";
    auto prevw = oss.width(6);
    auto prevf = oss.fill('0');
    oss << i;
    oss.width(prevw);
    oss.fill(prevf);

    const std::string path = oss.str();
    std::ofstream ofs(path, std::ios::out);

    ofs << "t C(t)\n";
    const std::vector<double> Cpi = staggered::C_pion<Group>(U, m, S.solver, S.tolerance_cg, S.solver_verbosity, S.seed_pf);
    for (size_t i = 0; i < U.getLt(); i++) {
      ofs << std::scientific << std::setprecision(16) << i << " " << Cpi[i] << "\n";
    }
    ofs.close();

    return;
  }
  
  /**
   * measures the planar wilson loops in temporal and spacial direction
   * -(0,1) loops for temporal, (1,2) and (1,3) loops for spacial
   * writes one line per configuration into the resultfiles. 
   * At the moment, measurements are implemented for dim=3,4.
   * @param U holds the gauge-configuration whose loops are measured
   * @param pparams holds information about the size of the lattice
   * @param sizeWloops: maximum extent up to which loops are measured
   * @param filenames: files into which the results of the measurements are written
   * @param i: index of the configuration
   * **/
  template <class Group>
  void
  meas_loops_planar_pot(const gaugeconfig<Group> &U,
                        const global_parameters::physics &pparams,
                        const double &sizeWloops,
                        const std::string &filename_coarse,
                        const std::string &filename_fine, 
                        const size_t &i){
    double loop;
    std::ofstream resultfile;
    //~ //calculate wilsonloops for potential
    if(pparams.ndims == 4){
      resultfile.open(filename_fine, std::ios::app);
      for (size_t t = 1 ; t <= pparams.Lt*sizeWloops ; t++){
        for (size_t x = 1 ; x <= pparams.Lx*sizeWloops ; x++){
          loop  = wilsonloop_non_planar(U, {t, x, 0, 0});
          resultfile << std::setw(14) << std::scientific << loop/U.getVolume() << "  " ;
        }
      }
      resultfile << i;
      resultfile << std::endl; 
      resultfile.close();
      
      resultfile.open(filename_coarse, std::ios::app);
      for (size_t y = 1 ; y <= pparams.Ly*sizeWloops ; y++){
        for (size_t x = 1 ; x <= pparams.Lx*sizeWloops ; x++){
          loop  = wilsonloop_non_planar(U, {0, x, y, 0});
          loop += wilsonloop_non_planar(U, {0, x, 0, y});
          resultfile << std::setw(14) << std::scientific << loop/U.getVolume()/2.0 << "  " ;
        }
      }
      resultfile << i;
      resultfile << std::endl; 
      resultfile.close();
    }
    if(pparams.ndims == 3){
      resultfile.open(filename_fine, std::ios::app);
      for (size_t t = 1 ; t <= pparams.Lt*sizeWloops ; t++){
        for (size_t x = 1 ; x <= pparams.Lx*sizeWloops ; x++){
          loop  = wilsonloop_non_planar(U, {t, x, 0});
          //~ loop  += wilsonloop_non_planar(U, {t, 0, x});
          resultfile << std::setw(14) << std::scientific << loop/U.getVolume() << "  " ;
        }
      }
      resultfile << i;
      resultfile << std::endl; 
      resultfile.close();
      
      resultfile.open(filename_coarse, std::ios::app);
      for (size_t y = 1 ; y <= pparams.Ly*sizeWloops ; y++){
        for (size_t x = 1 ; x <= pparams.Lx*sizeWloops ; x++){
          loop  = wilsonloop_non_planar(U, {0, x, y});
          //~ loop += wilsonloop_non_planar(U, {0, y, x});
          resultfile << std::setw(14) << std::scientific << loop/U.getVolume() << "  " ;
        }
      }
      resultfile << i;
      resultfile << std::endl; 
      resultfile.close();
    }
  }
  
  /**
   * measures the nonplanar wilson loops in temporal and spacial direction
   * writes one line per configuration into the resultfiles. 
   * At the moment, measurements are implemented for dim=3.
   * all possible loops (t,x,y) with x,y <=min(4, Lx), t<Lt*sizeWloops are measured.
   * @param U holds the gauge-configuration whose loops are measured
   * @param pparams holds information about the size of the lattice
   * @param sizeWloops: maximum extent up to which loops are measured
   * @param filenames: files into which the results of the measurements are written
   * @param i: index of the configuration
   * **/
  template <class Group>
  void
  meas_loops_nonplanar_pot(const gaugeconfig<Group> &U,
                        const global_parameters::physics &pparams,
                        const double &sizeWloops,
                        const std::string &filename_nonplanar,
                        const size_t &i){
    double loop;
    std::ofstream resultfile;
    size_t maxsizenonplanar = (pparams.Lx < 4) ? pparams.Lx : 4;
    resultfile.open(filename_nonplanar, std::ios::app);
    for (size_t t = 0 ; t <= pparams.Lt*sizeWloops ; t++){
      for (size_t x = 0 ; x <= maxsizenonplanar ; x++){
        for (size_t y = 0 ; y <= maxsizenonplanar ; y++){
        loop  = wilsonloop_non_planar(U, {t, x, y});
        resultfile << std::setw(14) << std::scientific << loop/U.getVolume() << "  " ;
        }
      }
    }
    resultfile << i;
    resultfile << std::endl; 
    resultfile.close();
  }
  
  
  using Complex = std::complex<double>;
  
  /**
   * measures average of spatial plaquettes and spatial-temporal plaquettes for each timeslice
   * **/
  template <class Group>
  void
  meas_one_time(const gaugeconfig<Group> &U,
                        const global_parameters::physics &pparams,
                        const std::string &filename_glueball,
                        const size_t &i){
    std::ofstream resultfile;
    resultfile.open(filename_glueball + "pure", std::ios::app);
    Complex timeslice;
    if(pparams.ndims==4){
      for (size_t t = 0 ; t < pparams.Lt ; t++){ 
          //spacial-spacial
        timeslice=operators::plaquette_one_timeslice(U, t, 1, 2);
        timeslice+=operators::plaquette_one_timeslice(U, t, 1, 3);
        timeslice+=operators::plaquette_one_timeslice(U, t, 2,3);
        timeslice/=(3*pparams.Lx*pparams.Ly*pparams.Lz); 
        resultfile << std::setw(14) << std::scientific << std::real(timeslice) << "  " << std::imag(timeslice) << "  " ;
          //spacial-temporal
        timeslice=operators::plaquette_one_timeslice(U, t, 0, 1);
        timeslice+=operators::plaquette_one_timeslice(U, t, 0, 2);
        timeslice+=operators::plaquette_one_timeslice(U, t, 0, 3);
        timeslice/=(3*pparams.Lx*pparams.Ly*pparams.Lz); 
        resultfile << std::setw(14) << std::scientific << std::real(timeslice) << "  " << std::imag(timeslice) << "  " ;
      }
    }
    if(pparams.ndims==3){
      for (size_t t = 0 ; t < pparams.Lt ; t++){ 
          //spacial-spacial
        timeslice=operators::plaquette_one_timeslice(U, t, 1, 2);
        timeslice/=(pparams.Lx*pparams.Ly*pparams.Lz); 
        resultfile << std::setw(14) << std::scientific << std::real(timeslice) << "  " << std::imag(timeslice) << "  " ;
        //~ std::cout << t << " " << std::setw(14) << std::scientific << std::real(timeslice) << "  " << std::imag(timeslice) << std::endl;
          //~ //spacial-temporal
        timeslice=operators::plaquette_one_timeslice(U, t, 0, 1);
        timeslice+=operators::plaquette_one_timeslice(U, t, 0, 2);
        timeslice/=(2*pparams.Lx*pparams.Ly*pparams.Lz); 
        resultfile << std::setw(14) << std::scientific << std::real(timeslice) << "  " << std::imag(timeslice) << "  " ;
      }
    }
    resultfile << i;
    resultfile << std::endl; 
    resultfile.close();
  }


  /**
   * measures average of spatial plaquettes and spatial-temporal plaquettes for each timeslice
   * **/
  template <class Group>
  void
  meas_one_time_v2(const gaugeconfig<Group> &U,
                        const global_parameters::physics &pparams,
                        const std::string &filename_glueball,
                        const size_t &i){
    std::ofstream resultfile;
    std::ostringstream resname;
    resname << filename_glueball << "proj." << i;
    resultfile.open(resname.str(), std::ios::out);
    std::vector<double> timeslice;
    std::vector<double> source;
      resultfile << "# t C_PC(t) ++ +- -+ -- sum_x P_ss, PC(t) ++ +- -+ --" << std::endl;
    for (size_t t = 0 ; t < pparams.Lt ; t++){ 
        //spacial-spacial
      timeslice=zerovector(4);
      for(size_t dim1=1; dim1 < pparams.ndims-1 ; dim1++){
        for(size_t dim2=dim1+1; dim2 < pparams.ndims ; dim2++){
          timeslice+=operators::measure_arbitrary_loop_one_timeslice_PC(U, t, /*lengths=*/{1,1,1,1}, /*directions=*/{dim1, dim2, dim1, dim2}, /*sign=*/{true, true, false, false});
        }
      }
      timeslice/=((pparams.ndims-1)*(pparams.ndims-2)/2.0*pparams.Lx*pparams.Ly*pparams.Lz);
      if(t==0){ 
          source=timeslice;
      } 
      resultfile << t << " " << std::setw(14) << std::scientific << //std::real(timeslice) << "  " << std::imag(timeslice) << "  " ;
        timeslice[0]*source[0] << " " << timeslice[1]*source[1] << " " << timeslice[2]*source[2] << " " << timeslice[3]*source[3] << " ";
      resultfile << std::setw(14) << std::scientific << //std::real(timeslice) << "  " << std::imag(timeslice) << "  " ;
        timeslice[0] << " " << timeslice[1] << " " << timeslice[2] << " " << timeslice[3];
          //spacial-temporal
      //~ for(size_t dim1=1; dim1 < pparams.ndims-1 ; dim1++){
        //~ timeslice=operators::measure_arbitrary_loop_one_timeslice_PC(U, t, /*lengths=*/{1,1,1,1}, /*directions=*/{dim1, 0, dim1, 0}, /*sign=*/{true, true, false, false});
      //~ }
      //~ timeslice/=(double)((pparams.ndims-1)*pparams.Lx*pparams.Ly*pparams.Lz); 
      //~ resultfile << std::setw(14) << std::scientific << //std::real(timeslice) << "  " << std::imag(timeslice) << "  " ;
        //~ timeslice[0] << " " << timeslice[1] << " " << timeslice[2] << " " << timeslice[3] << " ";
    resultfile << std::endl;
    }
    resultfile << "# config no." << i << std::endl; 
    resultfile.close();
  }


} // namespace omeasurements
