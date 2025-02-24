#include <iostream>
int main(int argc, char *argv[]) {
  int amount;

  std::cin >> amount;

  int books = amount * 0.75;

  amount -= books;

  int rest = (amount - books) / 3;

  int coffee = rest / 2;
  int flask = rest / 4;
  int metro = rest / 3;

  std::cout << books << std::endl;
  std::cout << coffee << std::endl;
  std::cout << flask << std::endl;
  std::cout << metro << std::endl;

  return 0;
}
