#pragma once
#include"gaugeconfig.hh"
#include<string>

void gradient_flow(gaugeconfig &U, std::string const &path, const double tmax = 9.99);
