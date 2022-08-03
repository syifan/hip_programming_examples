#include <pthread.h>
#include <stdio.h>

struct thread_args {
  int count;
  const char *message;
};

void *print_message(void *args) {
  struct thread_args *p = (struct thread_args *)args;

  for (int i = 0; i < p->count; i++) {
    printf("%s\n", p->message);
  }

  return NULL;
}

int main() {
  pthread_t thread1, thread2;

  struct thread_args args1 = {10, "Hello World From Thread 1!"};
  struct thread_args args2 = {15, "Hello World From Thread 2!"};

  pthread_create(&thread1, NULL, print_message, &args1);
  pthread_create(&thread2, NULL, print_message, &args2);

  pthread_join(thread1, NULL);
  pthread_join(thread2, NULL);

  return 0;
}
