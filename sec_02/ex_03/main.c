#include <pthread.h>
#include <stdio.h>

int total_count = 0;
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

struct accumulate_args {
  int count;
};

void *accumulate(void *args) {
  struct accumulate_args *p = (struct accumulate_args *)args;
  for (int i = 0; i < p->count; i++) {
    pthread_mutex_lock(&mutex);
    total_count++;
    pthread_mutex_unlock(&mutex);
  }

  return NULL;
}

int main() {
  pthread_t thread1, thread2;

  struct accumulate_args args1 = {10000};
  struct accumulate_args args2 = {15000};

  pthread_create(&thread1, NULL, accumulate, &args1);
  pthread_create(&thread2, NULL, accumulate, &args2);

  pthread_join(thread1, NULL);
  pthread_join(thread2, NULL);

  printf("Total count is %d\n", total_count);

  return 0;
}
