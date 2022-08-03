#include <cstdio>
#include <string>
#include <thread>

void print_message(int count, std::string message) {
  for (int i = 0; i < count; i++) {
    printf("%s\n", message.c_str());
  }
}

int main() {
  std::thread thread1(print_message, 10, "Hello, World!");
  std::thread thread2(print_message, 15, "Hello, World!");

  thread1.join();
  thread2.join();

  return 0;
}
