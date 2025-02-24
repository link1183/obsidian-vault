
#include <iostream>
int main() {
  int const BASE = 4;
  float fromage = 800.0;
  float eau = 2.0;
  float ail = 2.0;
  float pain = 400.0;

  int nb;

  std::cout << "Nb: ";
  std::cin >> nb;

  std::cout << fromage * nb / BASE << std::endl;
  std::cout << eau * nb / BASE << std::endl;
  std::cout << ail * nb / BASE << std::endl;
  std::cout << pain * nb / BASE << std::endl;
}
