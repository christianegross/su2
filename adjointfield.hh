#pragma once

#include"su2.hh"
#include"u1.hh"
#include<vector>
#include<random>
#include<cassert>
#include<cmath>

template<typename Float> class adjointsu2 {
public:
  adjointsu2(Float _a, Float _b, Float _c) : a(_a), b(_b), c(_c) {}
  adjointsu2() : a(0.), b(0.), c(0.) {}
  void flipsign() {
    a = -a;
    b = -b;
    c = -c;
  }
  Float geta() const {
    return a;
  }
  Float getb() const {
    return b;
  }
  Float getc() const {
    return c;
  }
  void seta(Float _a) {
    a = _a;
  }
  void setb(Float _a) {
    b = _a;
  }
  void setc(Float _a) {
    c = _a;
  }
  adjointsu2<Float> round(size_t n) const {
    Float dn = n;
    return adjointsu2(std::round(a * dn) / dn, std::round(b * dn) / dn, std::round(c * dn) / dn);
  }
  void operator=(const adjointsu2 &A) {
    a = A.geta();
    b = A.getb();
    c = A.getc();
  }
  void operator+=(const adjointsu2 &A) {
    a += A.geta();
    b += A.getb();
    c += A.getc();
  }
  void operator-=(const adjointsu2 &A) {
    a -= A.geta();
    b -= A.getb();
    c -= A.getc();
  }
  
private:
  Float a, b, c;
};

template<typename Float=double> inline adjointsu2<Float> get_deriv(su2 & A) {
  const Complex a = A.geta(), b = A.getb();
  return(adjointsu2<Float>(2.*std::imag(b), 2.*std::real(b), 2.*std::imag(a)));
}

template<typename Float> class adjointu1 {
public:
  adjointu1(Float _a) : a(_a) {}
  adjointu1() : a(0.) {}
  void flipsign() {
    a = -a;
  }
  Float geta() const {
    return a;
  }
  void seta(Float _a) {
    a = _a;
  }
  adjointu1<Float> round(size_t n) const {
    Float dn = n;
    return adjointu1(std::round(a * dn) / dn);
  }
  void operator=(const adjointu1 &A) {
    a = A.geta();
  }
  void operator+=(const adjointu1 &A) {
    a += A.geta();
  }
  void operator-=(const adjointu1 &A) {
    a -= A.geta();
  }
  
private:
  Float a;
};

// The following class will be used to deliver the
// adjoint type depending on the gauge group
template<typename Float, class Group> struct adjoint_type {
  typedef Group type;
};

template<typename Float> struct adjoint_type<Float, su2> {
  typedef adjointsu2<Float> type;
};

template<typename Float> struct adjoint_type<Float, _u1> {
  typedef adjointu1<Float> type;
};


template<typename Float, class Group=su2> class adjointfield {
public:
  //  typedef typename accum_type<T>::type ;
  //  using value_type = adjoint<Float>;
  using value_type = typename adjoint_type<Float, Group>::type;
  adjointfield(const size_t Ls, const size_t Lt) : 
    Ls(Ls), Lt(Lt), volume(Ls*Ls*Ls*Lt) {
    data.resize(volume*4);
  }
  adjointfield(const adjointfield &U) :
    Ls(U.getLs()), Lt(U.getLt()), volume(U.getVolume()) {
    data.resize(volume*4);
    for(size_t i = 0; i < getSize(); i++) {
      data[i] = U[i];
    }
  }
  void flipsign() {
    for(size_t i = 0; i < getSize(); i++) {
      data[i].flipsign();
    }
  }
  size_t storage_size() const { return data.size() * sizeof(value_type); };
  size_t getLs() const {
    return(Ls);
  }
  size_t getLt() const {
    return(Lt);
  }
  size_t getVolume() const {
    return(volume);
  }
  size_t getSize() const {
    return(volume*4);
  }
  void operator=(const adjointfield &U) {
    Ls = U.getLs();
    Lt = U.getLt();
    volume = U.getVolume();
    data.resize(U.getSize());
    for(size_t i = 0; i < U.getSize(); i++) {
      data[i] = U[i];
    }
  }

  value_type &operator()(size_t const t, size_t const x, size_t const y, size_t const z, size_t const mu) {
    return data[ getIndex(t, x, y, z, mu) ];
  }

  const value_type &operator()(size_t const t, size_t const x, size_t const y, size_t const z, size_t const mu) const {
    return data[ getIndex(t, x, y, z, mu) ];
  }

  value_type &operator()(std::vector<size_t> const &coords, size_t const mu) {
    return data[ getIndex(coords[0], coords[1], coords[2], coords[3], mu) ];
  }

  const value_type &operator()(std::vector<size_t> const &coords, size_t const mu) const {
    return data[ getIndex(coords[0], coords[1], coords[2], coords[3], mu) ];
  }

  value_type &operator[](size_t const index) {
    return data[ index ];
  }

  const value_type &operator[](size_t const index) const {
    return data[ index ];
  }

private:
  size_t Ls, Lt, volume;
  
  std::vector<value_type> data;
  
  size_t getIndex(const size_t t, const size_t x, const size_t y, const size_t z, const size_t mu) const {
    size_t y0 = (t + Lt) % Lt;
    size_t y1 = (x + Ls) % Ls;
    size_t y2 = (y + Ls) % Ls;
    size_t y3 = (z + Ls) % Ls;
    size_t _mu = (mu + 4) % 4;
    return( (((y0*Ls + y1)*Ls + y2)*Ls + y3)*4 + _mu );
  }
};


template<typename Float>  adjointsu2<Float> operator*(const Float &x, const adjointsu2<Float> &A) {
  adjointsu2<Float> res;
  res.seta(x * A.geta());
  res.setb(x * A.getb());
  res.setc(x * A.getc());
  return res;
}

template<typename Float>  adjointu1<Float> operator*(const Float &x, const adjointu1<Float> &A) {
  adjointu1<Float> res;
  res.seta(x * A.geta());
  return res;
}


template<typename Float>  adjointfield<Float> operator*(const Float &x, const adjointfield<Float> &A) {
  adjointfield<Float> res(A.getLs(), A.getLt());
  for(size_t i = 0; i < A.getSize(); i++) {
    res[i].seta( x * A[i].geta());
    res[i].setb( x * A[i].getb());
    res[i].setc( x * A[i].getc());
  }
  return res;
}

template<typename Float> Float operator*(const adjointfield<Float> &A, const adjointfield<Float> &B) {
  Float res = 0.;
  assert(A.getSize() == B.getSize());
  for(size_t i = 0; i < A.getSize(); i++) {
    res += A[i].geta()*B[i].geta() + A[i].getb()*B[i].getb() + A[i].getc()*B[i].getc();
  }
  return res;
}

template<class URNG, typename Float> adjointfield<Float> initnormal(URNG &engine, size_t Ls, size_t Lt) {
  adjointfield<Float> A(Ls, Lt);
  std::normal_distribution<double> normal(0., 1.);
  for(size_t i = 0; i < A.getSize(); i++) {
    A[i].seta(Float(normal(engine)));
    A[i].setb(Float(normal(engine)));
    A[i].setc(Float(normal(engine)));
  }
  return A;
}

template<typename Float> void zeroadjointfield(adjointfield<Float> &A) {
  for(size_t i = 0; i < A.getSize(); i++) {
    A[i].seta(0.);
    A[i].setb(0.);
    A[i].setc(0.);
  }
}

