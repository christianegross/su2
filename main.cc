#include"su2.hh"
#include"gaugeconfig.hh"
#include"gauge_energy.hh"
#include"random_gauge_trafo.hh"
#include"sweep.hh"
#include"wilsonloop.hh"
#include"gradient_flow.hh"
#include"print_program_options.hh"

#include<iostream>
#include<sstream>
#include<vector>
#include<random>
#include<boost/program_options.hpp>

using std::vector;
using std::cout;
using std::endl;
namespace po = boost::program_options;

int main(int ac, char* av[]) {
  size_t Ls = 8, Lt = 16;
  double beta = 2.3;
  size_t N_hit = 10;
  size_t N_meas = 2000;
  double delta = 0.1;
  double heat = 0.;
  size_t N_save = 200;
  int seed = 13526463;
  
  
  try {
    po::options_description desc("Allowed options");
    desc.add_options()
      ("help,h", "produce this help message")
      ("spatialsize,L", po::value<size_t>(&Ls), "spatial lattice size")
      ("temporalsize,T", po::value<size_t>(&Lt), "temporal lattice size")
      ("nhit", po::value<size_t>(&N_hit)->default_value(10), "N_hit")
      ("nsave", po::value<size_t>(&N_save)->default_value(1000), "N_save")
      ("nmeas,n", po::value<size_t>(&N_meas)->default_value(10), "total number of sweeps")
      ("beta,b", po::value<double>(&beta), "beta value")
      ("seed,s", po::value<int>(&seed)->default_value(13526463), "PRNG seed")
      ("heat", po::value<double>(&heat)->default_value(1.), "randomness of the initial config, 1: hot, 0: cold")
      ("delta,d", po::value<double>(&delta), "delta")
      ;

    po::variables_map vm;
    po::store(po::parse_command_line(ac, av, desc), vm);
    po::notify(vm);
    
    if (vm.count("help")) {
      cout << desc << endl;
      return 1;
    }
    if (!vm.count("spatialsize") && !vm.count("help")) {
      std::cerr << "spatial lattice size must be given!" << endl;
      cout << endl << desc << endl;
      return 1;
    }
    if (!vm.count("temporalsize") && !vm.count("help")) {
      std::cerr << "temporal lattice size must be given!" << endl;
      cout << endl << desc << endl;
      return 1;
    }
    if (!vm.count("beta") && !vm.count("help")) {
      std::cerr << "beta value must be specified!" << endl;
      cout << endl << desc << endl;
      return 1;
    }
    PrintVariableMap(vm);
  }
  catch(std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  catch(...) {
    std::cerr << "Exception of unknown type!\n";
  }


  

  gaugeconfig U(Ls, Lt, beta);
  U = hotstart(Ls, Lt, 123456, heat);
  //U = coldstart(Ls, Lt);
  

  double plaquette = gauge_energy(U);
  cout << "Initital Plaquette: " << plaquette/U.getVolume()/N_c/6. << endl; 

  random_gauge_trafo(U, 654321);
  plaquette = gauge_energy(U);
  cout << "Plaquette after rnd trafo: " << plaquette/U.getVolume()/N_c/6. << endl; 

  double rate = 0.;
  for(size_t i = 0; i < N_meas; i++) {
    std::mt19937 engine(seed+i);
    rate += sweep(U, engine, delta, N_hit, beta);
    cout << i << " " << gauge_energy(U)/U.getVolume()/N_c/6. << endl;
    if(i > 0 && (i % N_save) == 0) {
      std::ostringstream os;
      os << "config." << Ls << "." << Lt << ".b" << beta << "." << i << std::ends;
      U.save(os.str());
    }
  }
  cout << rate/static_cast<double>(N_meas) << endl;

  std::ostringstream os;
  os << "config." << Ls << "." << Lt << ".b" << U.getBeta() << ".final" << std::ends;
  U.save(os.str());

  return(0);
}

