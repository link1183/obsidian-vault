#include <cmath>
#include <iostream>
#include <ostream>
#include <string>

void ex1();
void ex2();
void ex3();
void ex4();
void ex5();

int main(int argc, char *argv[]) {
  // ex1();
  // ex2();
  // ex3();
  // ex4();
  ex5();
  return 0;
}

void ex1() {
  std::cout << "\n\nex1" << std::endl;

  int num;
  std::cin >> num;

  std::string isEven = num % 2 == 0 ? "even" : "odd";

  if (num > 0) {
    std::cout << "positive " << isEven << std::endl;
  } else if (num < 0) {
    std::cout << "negative " << isEven << std::endl;
  } else {
    std::cout << "zero " << isEven << std::endl;
  }
}

void ex2() {
  std::cout << "\n\nex2" << std::endl;

  double x;
  std::cin >> x;

  if (x < -1 || x >= 1) {
    std::cout << "no" << std::endl;
  } else {
    std::cout << "yes" << std::endl;
  }
}

void ex3() {
  std::cout << "\n\nex3" << std::endl;

  double x;
  std::cin >> x;

  std::cout << "fuckit" << std::endl;
}

void ex4() {
  double x;

  std::cin >> x;

  double eq1 = x / (1 - exp(x));
  double eq2 = x * log(x) * exp(2 / (x - 1));
  double eq3 = (-x - std::sqrt(pow(x, 2) - 8 * x)) / (2 - x);
  double eq4 = std::sqrt((sin(x) - (x / 20)) * (log(pow(x, 2) - (1 / x))));

  std::cout << "1 : " << eq1 << std::endl;
  std::cout << "2 : " << eq2 << std::endl;
  std::cout << "3 : " << eq3 << std::endl;
  std::cout << "4 : " << eq4 << std::endl;
}

void ex5() {
  double a0;
  double a1;
  double a2;

  std::cout << "a0 ";
  std::cin >> a0;
  std::cout << "a1 ";
  std::cin >> a1;
  std::cout << "a2 ";
  std::cin >> a2;

  double Q = (3 * a1 - pow(a2, 2)) / 9;
  double R = (9 * a2 * a1 - 27 * a0 - 2 * pow(a2, 3)) / 54;

  double D = pow(Q, 3) + pow(R, 2);

  if (D < 0) {
    double theta = std::acos(R / std::sqrt(-pow(Q, 3)));
    double z1 = 2 * std::sqrt(-Q) * cos(theta / 3) - 1.0 / 3.0 * a2;
    double z2 =
        2 * std::sqrt(-Q) * cos((theta + 2 * M_PI) / 3) - 1.0 / 3.0 * a2;
    double z3 =
        2 * std::sqrt(-Q) * cos((theta + 4 * M_PI) / 3) - 1.0 / 3.0 * a2;

    std::cout << z1 << std::endl;
    std::cout << z2 << std::endl;
    std::cout << z3 << std::endl;
    return;
  } else {
    double S = std::cbrt(R + std::sqrt(D));
    double T = std::cbrt(R - std::sqrt(D));

    if (D == 0 && S + T != 0) {
      double z1 = -1.0 / 3.0 * a2 + (S + T);
      double z2 = -(S + T) / 2 - a2 / 3;
      std::cout << z1 << std::endl;
      std::cout << z2 << std::endl;
      return;
    } else {
      double z1 = -1.0 / 3.0 * a2 + (S + T);

      std::cout << z1 << std::endl;
      return;
    }
  }
}
