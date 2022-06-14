// measure-u1.cc
/**
 * @file measure-u1.cc
 * @author Carsten Urbach (urbach@hiskp.uni-bonn.de)
 * @author Simone Romiti (simone.romiti@uni-bonn.de)
 * @brief offline measurements of observables over previously generate gauge
 * configurations
 * @version 0.1
 * @date 2022-05-11
 *
 * @copyright Copyright (c) 2022
 *
 */

#include "energy_density.hh"
#include "flat-gauge_energy.hpp"
#include "gaugeconfig.hh"
#include "io.hh"
#include "md_update.hh"
#include "monomial.hh"
#include "omeasurements.hpp"
#include "parse_input_file.hh"
#include "polyakov_loop.hh"
#include "random_gauge_trafo.hh"
#include "smearape.hh"
#include "su2.hh"
#include "u1.hh"
#include "version.hh"

#include <algorithm>
#include <boost/program_options.hpp>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>

namespace po = boost::program_options;

#include <boost/filesystem.hpp>

int main(int ac, char *av[]) {
  std::cout << "## Measuring Tool for U(1) gauge theory" << std::endl;
  std::cout << "## (C) Carsten Urbach <urbach@hiskp.uni-bonn.de> (2017)" << std::endl;
  std::cout << "## GIT branch " << GIT_BRANCH << " on commit " << GIT_COMMIT_HASH
            << std::endl;

  namespace gp = global_parameters;
  gp::physics pparams; // physics parameters
  gp::measure_u1 mparams; // measure parameters

  std::string input_file; // yaml input file path
  int err = input_file_parsing::parse_command_line(ac, av, input_file);
  if (err > 0) {
    return err;
  }

  namespace in_meas = input_file_parsing::u1::measure;
  err = in_meas::parse_input_file(input_file, pparams, mparams);
  if (err > 0) {
    return err;
  }

  boost::filesystem::create_directories(boost::filesystem::absolute(mparams.conf_dir));
  boost::filesystem::create_directories(boost::filesystem::absolute(mparams.resdir));

  gaugeconfig<_u1> U(pparams.Lx, pparams.Ly, pparams.Lz, pparams.Lt, pparams.ndims,
                     pparams.beta);

  // get basename for configs
  std::string conf_path_basename = io::get_conf_path_basename(pparams, mparams);

  // filename needed for saving results from potential and potentialsmall
  const std::string filename_fine = io::measure::get_filename_fine(pparams, mparams);
  const std::string filename_coarse = io::measure::get_filename_coarse(pparams, mparams);
  const std::string filename_nonplanar =
    io::measure::get_filename_nonplanar(pparams, mparams);
  const std::string filename_glueball =
    io::measure::get_filename_glueball(pparams, mparams);

  // write explanatory headers into result-files, also check if measuring routine is
  // implemented for given dimension
  if (mparams.potentialplanar) {
    io::measure::set_header_planar(pparams, mparams, filename_coarse, filename_fine);
  }
  if (mparams.potentialnonplanar) {
    io::measure::set_header_nonplanar(pparams, mparams, filename_nonplanar);
  }
  if (mparams.glueball) {
    io::measure::set_header_glueball(pparams, mparams, filename_glueball);
  }

  /**
   * do the measurements themselves:
   * load each configuration, check for gauge invariance
   * if selected, measure Wilson-Loop and gradient flow
   * if chosen, do APE-smearing
   * if selected, then measure potential and small potential
   * the "small potential" are the nonplanar loops, but only with small extent in x, y
   * */

  const size_t istart =
    mparams.icounter == 0 ? mparams.icounter + mparams.nstep : mparams.icounter;
  for (size_t i = istart; i < mparams.n_meas * mparams.nstep + mparams.icounter;
       i += mparams.nstep) {
    std::string path_i = conf_path_basename + "." + std::to_string(i);
    int ierrU = U.load(path_i);
    if (ierrU == 1) { // cannot load gauge config
      continue;
    }

    const double ndims_fact =
      spacetime_lattice::num_pLloops_half(U.getndims()); // d*(d-1)/2
    double plaquette = flat_spacetime::gauge_energy(U);
    double density = 0., Q = 0.;
    energy_density(U, density, Q);
    std::cout << "## Initial Plaquette: "
              << plaquette / U.getVolume() / double(U.getNc()) / ndims_fact << std::endl;
    std::cout << "## Initial Energy density: " << density << std::endl;

    random_gauge_trafo(U, mparams.seed);
    plaquette = flat_spacetime::gauge_energy(U);
    energy_density(U, density, Q);
    std::cout << "## Plaquette after rnd trafo: " << std::scientific << std::setw(15)
              << plaquette / U.getVolume() / double(U.getNc()) / ndims_fact << std::endl;
    std::cout << "## Energy density: " << density << std::endl;

    if (mparams.Wloop) {
      omeasurements::meas_wilson_loop<_u1>(U, i, mparams.conf_dir);
    }

    if (mparams.gradient) {
      omeasurements::meas_gradient_flow<_u1>(U, i, mparams.conf_dir, mparams.tmax);
    }

    if (mparams.pion_staggered) {
      omeasurements::meas_pion_correlator<_u1>(U, i, pparams.m0, mparams);
    }

    if (mparams.potentialplanar || mparams.potentialnonplanar || mparams.glueball) {
      // smear lattice
      for (size_t smears = 0; smears < mparams.n_apesmear; smears += 1) {
        smearlatticeape(U, mparams.alpha, mparams.smear_spatial_only,
                        mparams.smear_temporal_only);
      }
      double loop;
      if (mparams.potentialplanar) {
        omeasurements::meas_loops_planar_pot(U, pparams, mparams.sizeWloops,
                                             filename_coarse, filename_fine, i);
      }

      if (mparams.potentialnonplanar) {
        omeasurements::meas_loops_nonplanar_pot(U, pparams, mparams.sizeWloops,
                                                filename_nonplanar, i);
      }

      if( mparams.glueball){
        omeasurements::meas_one_time_v2(U, pparams, filename_glueball, i);
        omeasurements::meas_one_time(U, pparams, filename_glueball, i);
        //~ omeasurements::meas_one_time(U, pparams, i);
      }
    }
  }

  return (0);
}
