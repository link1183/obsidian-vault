#include <stdio.h>
#include <stdlib.h>

void unreachableFunction(void) {
  printf("I'm hacked! I am a hidden function!\n");
  exit(0);
}

int main(void) {
  printf("Hello World!\n");

  return 0;
}
