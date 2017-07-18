#include"su2.hh"
#include"gaugeconfig.hh"
#include"gauge_energy.hh"
#include"random_gauge_trafo.hh"
#include"sweep.hh"
#include"wilsonloop.hh"
#include"gradient_flow.hh"

#include<iostream>
#include<sstream>
#include<vector>

using std::vector;
using std::cout;
using std::endl;


int main() {
  const size_t Ls = 8, Lt = 8;
  const double beta = 2.3;
  const size_t N_hit = 10;
  const size_t N_meas = 2000;
  const double delta = 0.1;
  const size_t N_save = 20;
  gaugeconfig U(Ls, Lt, beta);
  U = hotstart(Ls, Lt, 123456, 0.1);
  //U = coldstart(Ls, Lt);
  
  double plaquette = gauge_energy(U);
  cout << "Initital Plaquette: " << plaquette/U.getVolume()/N_c/6. << endl; 

  random_gauge_trafo(U, 654321);
  plaquette = gauge_energy(U);
  cout << "Plaquette after rnd trafo: " << plaquette/U.getVolume()/N_c/6. << endl; 

  double rate = 0.;
  for(size_t i = 0; i < N_meas; i++) {
    rate += sweep(U, 13243546, delta, N_hit, beta);
    cout << i << " " << gauge_energy(U)/U.getVolume()/N_c/6. << endl;
    if(i > 0 && (i % N_save) == 0) {
      std::ostringstream os;
      os << "config." << Ls << "." << Lt << "." << beta << "." << i << std::ends;
      U.save(os.str());
    }
  }
  cout << rate/static_cast<double>(N_meas) << endl;

  std::ostringstream os;
  os << "config." << Ls << "." << Lt << "." << U.getBeta() << ".final" << std::ends;
  U.save(os.str());

  return(0);
}

