#include"su2.hh"
#include"u1.hh"
#include"gaugeconfig.hh"
#include"gauge_energy.hh"
#include"random_gauge_trafo.hh"
#include"sweep.hh"
#include"wilsonloop.hh"
#include"md_update.hh"
#include"monomial.hh"
#include"gradient_flow.hh"
#include"energy_density.hh"
#include"parse_commandline.hh"
#include"version.hh"
#include"smearape.hh"

#include<iostream>
#include<iomanip>
#include<sstream>
#include<vector>
#include<random>
#include<boost/program_options.hpp>

namespace po = boost::program_options;

using std::cout;
using std::endl;


int main(int ac, char* av[]) {
  general_params gparams;
  size_t nstep;
  bool Wloop;
  bool gradient;
  bool lyapunov;
  double tmax;
  size_t n_steps;
  size_t exponent;
  double tau;
  size_t integs;
  bool append;
  size_t n_apesmear;
  double alpha;
  bool smearspacial;
  bool potential;

  cout << "## Measuring Tool for U(1) gauge theory" << endl;
  cout << "## (C) Carsten Urbach <urbach@hiskp.uni-bonn.de> (2017)" << endl;
  cout << "## GIT branch " << GIT_BRANCH << " on commit " << GIT_COMMIT_HASH << endl << endl;  

  po::options_description desc("Allowed options");
  add_general_options(desc, gparams);
  // add measure specific options
  desc.add_options()
    ("Wloops", po::value<bool>(&Wloop)->default_value(false), "measure Wilson loops")
    ("gradient", po::value<bool>(&gradient)->default_value(false), "meausre Grandient flow")
    ("nstep", po::value<size_t>(&nstep)->default_value(1), "measure each nstep config")
    ("tmax", po::value<double>(&tmax)->default_value(9.99), "tmax for gradient flow")
    ("nsteps", po::value<size_t>(&n_steps)->default_value(1000), "n_steps")
    ("tau", po::value<double>(&tau)->default_value(1.), "trajectory length tau")
    ("exponent", po::value<size_t>(&exponent)->default_value(0), "exponent for rounding")
    ("integrator", po::value<size_t>(&integs)->default_value(0), "itegration scheme to be used: 0=leapfrog, 1=lp_leapfrog, 2=omf4, 3=lp_omf4")
    ("append", po::value<bool>(&append)->default_value(false), "are measurements appended to an existing file, or should it be overwritten?")
    ("napesmears", po::value<size_t>(&n_apesmear)->default_value(0), "number of APE smearings done on the lattice before measurement")
    ("apealpha", po::value<double>(&alpha)->default_value(1.0), "parameter alpha for APE smearings")
    ("spacialsmear", po::value<bool>(&smearspacial)->default_value(false), "should smearing be done only for spacial links?")
    ("potential", po::value<bool>(&potential)->default_value(false), "measure potential")
    ;

  int err = parse_commandline(ac, av, desc, gparams);
  if(err > 0) {
    return err;
  }
  

  gaugeconfig<_u1> U(gparams.Lx, gparams.Ly, gparams.Lz, gparams.Lt, gparams.ndims, gparams.beta);
 

  //~ open file for saving results
  std::ofstream resultfile;
  char filename[200];
  double loop; 
  
  if(potential) {
    //~ print heads of columns: W(r, t), W(x, y)
    if(!append){
    for (size_t x=1 ; x<=gparams.Lx/2 ; x++){
      if(gparams.ndims==4){
      sprintf(filename, "result.u1potential.Nt%lu.Ns%lu.b%f.xi%f.nape%lu.alpha%f.x%lu",gparams.Lt, gparams.Lx,U.getBeta(), gparams.xi, n_apesmear, alpha, x);
      }
      if(gparams.ndims==3){
      sprintf(filename, "result2p1d.u1potential.Nt%lu.Ns%lu.b%f.xi%f.nape%lu.alpha%f.x%lu",gparams.Lt, gparams.Lx,U.getBeta(), gparams.xi, n_apesmear, alpha, x);
      }
      resultfile.open(filename, std::ios::out);
      resultfile << "#";
      for (size_t t=1 ; t<=gparams.Lt/2 ; t++){
        resultfile << std::setw(5) << "W(x=" << std::setw(2) << x << ", t=" << std::setw(2) << t << ")  " ; 
        resultfile << std::setw(5) << "Wo(x=" << std::setw(2) << x << ", t=" << std::setw(2) << t << ")  " ; 
      }
      for (size_t y=1 ; y<=gparams.Ly/2 ; y++){
        resultfile << std::setw(5) << "W(x=" << std::setw(2) << x << ", y=" << std::setw(2) << y << ")  " ; 
        resultfile << std::setw(5) << "Wo(x=" << std::setw(2) << x << ", y=" << std::setw(2) << y << ")  " ; 
      }
      resultfile << std::endl; 
      resultfile.close();    
    }
    }
}

  for(size_t i = gparams.icounter; i < gparams.N_meas*nstep+gparams.icounter; i+=nstep) {
    std::ostringstream os;
    os << "configu1." << gparams.Lx << "." << gparams.Ly << "." << gparams.Lz << "." << gparams.Lt << ".b" << std::fixed << U.getBeta() << ".x" << gparams.xi << "." << i << std::ends;
    U.load(os.str());
    
    double plaquette = gauge_energy(U);
    double density = 0., Q=0.;
    energy_density(U, density, Q);
    cout << "## Initital Plaquette: " << plaquette/U.getVolume()/double(U.getNc())/6. << endl; 
    cout << "## Initial Energy density: " << density << endl;
    
    random_gauge_trafo(U, gparams.seed);
    plaquette = gauge_energy(U);
    energy_density(U, density, Q);
    cout << "## Plaquette after rnd trafo: " << std::scientific << std::setw(15) << plaquette/U.getVolume()/double(U.getNc())/6. << endl; 
    cout << "## Energy density: " << density << endl;
    
    if(Wloop) {
      std::ostringstream os;
      os << "wilsonloop.";
      auto prevw = os.width(6);
      auto prevf = os.fill('0');
      os << i;
      os.width(prevw);
      os.fill(prevf);
      os << ".dat" << std::ends;
      compute_spacial_loops(U, os.str());
      cout << endl;
    }
    if(gradient) {
      std::ostringstream os;
      os << "gradient_flow.";
      auto prevw = os.width(6);
      auto prevf = os.fill('0');
      os << i;
      os.width(prevw);
      os.fill(prevf);
      gradient_flow(U, os.str(), tmax);
    }
    if(potential) {
              //smear lattice
      for (size_t smears=0 ; smears<n_apesmear ; smears +=1){
        smearlatticeape(U, alpha, smearspacial);
      }
      //~ //calculate wilsonloops
      if(gparams.ndims==4){
      for (size_t x=1 ; x<=gparams.Lx/2 ; x++){
        sprintf(filename, "result.u1potential.Nt%lu.Ns%lu.b%f.xi%f.nape%lu.alpha%f.x%lu",gparams.Lt, gparams.Lx,U.getBeta(), gparams.xi, n_apesmear, alpha, x);
        resultfile.open(filename, std::ios::app);
      
      //Measure for two radii each time by changing one of the coordinates not needed for the measurement
      //Measure (x,t) and (x,y), with "t" the anisotropic direction, "x" the "first" isotropic direction and "y" taken as the average of the other two directions
        for (size_t t=1 ; t<=gparams.Lt/2 ; t++){
          loop=wilsonloop_non_planar(U, {t, x, 0, 0});
          loop+=wilsonloop_non_planar(U, {t, 0, x, 0});
          loop+=wilsonloop_non_planar(U, {t, 0, 0, x});
          resultfile << std::setw(14) << std::scientific << loop/U.getVolume()/3.0 << "  " ; 
          
          loop= wilsonloop_non_planar(U, {t, x, 1, 0});
          loop+=wilsonloop_non_planar(U, {t, x, 0, 1});
          loop+=wilsonloop_non_planar(U, {t, 1, x, 0});
          loop+=wilsonloop_non_planar(U, {t, 0, x, 1});
          loop+=wilsonloop_non_planar(U, {t, 0, 1, x});
          loop+=wilsonloop_non_planar(U, {t, 1, 0, x});
          resultfile << std::setw(14) << std::scientific << loop/U.getVolume()/6.0 << "  " ;
        }
        for (size_t y=1 ; y<=gparams.Lx/2 ; y++){ 
          loop= wilsonloop_non_planar(U, {0, x, y, 0});
          loop+=wilsonloop_non_planar(U, {0, x, 0, y});
          loop+=wilsonloop_non_planar(U, {0, y, x, 0});
          loop+=wilsonloop_non_planar(U, {0, 0, x, y});
          loop+=wilsonloop_non_planar(U, {0, 0, y, x});
          loop+=wilsonloop_non_planar(U, {0, y, 0, x});
          resultfile << std::setw(14) << std::scientific << loop/U.getVolume()/6.0 << "  " ;
          
          loop= wilsonloop_non_planar(U, {1, x, y, 0});
          loop+=wilsonloop_non_planar(U, {1, x, 0, y});
          loop+=wilsonloop_non_planar(U, {1, y, x, 0});
          loop+=wilsonloop_non_planar(U, {1, 0, x, y});
          loop+=wilsonloop_non_planar(U, {1, 0, y, x});
          loop+=wilsonloop_non_planar(U, {1, y, 0, x});
          resultfile << std::setw(14) << std::scientific << loop/U.getVolume()/6.0 << "  " ;
        }
      resultfile << std::endl;
      resultfile.close(); 
      }
      }
      if(gparams.ndims==3){
      for (size_t x=1 ; x<=gparams.Lx/2 ; x++){
        sprintf(filename, "result2p1d.u1potential.Nt%lu.Ns%lu.b%f.xi%f.nape%lu.alpha%f.x%lu",gparams.Lt, gparams.Lx,U.getBeta(), gparams.xi, n_apesmear, alpha, x);
        resultfile.open(filename, std::ios::app);
      
      //Measure for two radii each time by changing one of the coordinates not needed for the measurement
      for (size_t t=1 ; t<=gparams.Lt/2 ; t++){
          loop=wilsonloop_non_planar(U, {t, x, 0});
          loop+=wilsonloop_non_planar(U, {t, 0, x});
          resultfile << std::setw(14) << std::scientific << loop/U.getVolume()/2.0 << "  " ; 
          
          loop= wilsonloop_non_planar(U, {t, x, 1});
          loop+=wilsonloop_non_planar(U, {t, 1, x});
          resultfile << std::setw(14) << std::scientific << loop/U.getVolume()/2.0 << "  " ;
        }
        for (size_t y=1 ; y<=gparams.Lx/2 ; y++){ 
          loop= wilsonloop_non_planar(U, {0, x, y});
          loop+=wilsonloop_non_planar(U, {0, y, x});
          resultfile << std::setw(14) << std::scientific << loop/U.getVolume()/2.0 << "  " ;
          
          loop= wilsonloop_non_planar(U, {1, x, y});
          loop+=wilsonloop_non_planar(U, {1, y, x});
          resultfile << std::setw(14) << std::scientific << loop/U.getVolume()/2.0 << "  " ;
        }
      resultfile << std::endl;
      resultfile.close(); 
      }
      }
          
    }
  }

  return(0);
}
