#include <iostream>

void ex1();

int main(int argc, char *argv[]) {
  ex1();
  return 0;
}

void ex1() {
  int age;

  std::cout << "Enter your age: ";
  std::cin >> age;
  std::cout << "Your age is: " << age << std::endl;
}
