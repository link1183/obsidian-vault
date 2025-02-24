#include <iostream>
int main(int argc, char *argv[]) {
  float x, y;
  float a, b, c, d;
  x = 2;
  y = 4;
  a = x + y;
  b = x - y;
  c = x * y;
  d = x / y;

  std::cout << a << " " << b << " " << c << " " << d << std::endl;
  return 0;
}
