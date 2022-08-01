#include <pthread.h>
#include <stdio.h>

void *print_message() {
  for (int i = 0; i < 10; i++) {
    printf("Hello World!\n");
  }

  return NULL;
}

int main() {
  pthread_t thread1, thread2;

  pthread_create(&thread1, NULL, print_message, NULL);
  pthread_create(&thread2, NULL, print_message, NULL);

  pthread_join(thread1, NULL);
  pthread_join(thread2, NULL);

  return 0;
}
