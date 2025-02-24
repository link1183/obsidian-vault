// https://iccsv.epfl.ch/series-prog/serie4.php

#include <atomic>
#include <cmath>
#include <initializer_list>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

std::mutex mtx;

void check_divisors(long long int n, long long int start, long long int end,
                    std::atomic<bool> &isPrime, std::atomic<int> &j) {
  for (long long int i = start; i > end && isPrime.load(); --i) {
    if (n % i == 0) {
      isPrime.store(false);
      j.store(i);
      return;
    }
  }
}

void ex1();
void ex2();
void ex3();
void ex4();
void ex5();
void ex6();
void ex7();
void ex8();

int main(int argc, char *argv[]) {
  ex5();
  return 0;
}

void ex1() {
  for (auto i : {2, 3, 4, 5, 6, 7, 8, 9, 10}) {
    std::cout << "table of " << i << std::endl;

    for (int j = 1; j <= 10; j++) {
      std::cout << "\t" << j << " * " << i << " = " << j * i << std::endl;
    }
  }
}

void ex2() {
  double const g = 9.81;

  double H0;
  double eps;
  int nbr;

  std::cin >> H0;
  std::cin >> eps;
  std::cin >> nbr;

  double v, v1, h1;
  double h = H0;

  for (int i = nbr; i > 0; i--) {
    v = std::sqrt(2 * h * g);
    v1 = eps * v;
    h1 = pow(v1, 2) / (2 * g);
    h = h1;
  }

  std::cout << h1 << std::endl;
}

void ex3() {
  double const g = 9.81;

  double H0;
  double eps;
  double hFin;

  std::cin >> H0;
  std::cin >> eps;
  std::cin >> hFin;

  double v, v1, h1;
  double h = H0;

  int count = 0;

  do {
    v = std::sqrt(2 * h * g);
    v1 = eps * v;
    h1 = pow(v1, 2) / (2 * g);
    h = h1;

    count++;
  } while (h1 > hFin);

  std::cout << count << std::endl;
}

void ex4() {
  long long int n;
  std::cin >> n;

  if (n % 2 == 0) {
    std::cout << "nop, 2" << std::endl;
    return;
  }

  std::atomic<bool> isPrime(true);
  std::atomic<int> j(0);
  int num_threads = std::thread::hardware_concurrency();
  std::vector<std::thread> threads;
  long long int chunk_size = (n - 2) / num_threads;

  for (int i = 0; i < num_threads; ++i) {
    long long int start = n - 1 - i * chunk_size;
    long long int end = (i == num_threads - 1) ? 1 : start - chunk_size;
    threads.emplace_back(check_divisors, n, start, end, std::ref(isPrime),
                         std::ref(j));
  }

  for (auto &t : threads) {
    t.join();
  }

  if (isPrime.load()) {
    std::cout << "yep" << std::endl;
  } else {
    std::cout << "nop, " << j.load() << std::endl;
  }
}

void ex5() {}
