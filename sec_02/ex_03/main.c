#include <pthread.h>
#include <stdio.h>

int total_count = 0;
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

struct thread_args {
  int count;
};

void *accumulate_mutex(void *args) {
  struct thread_args *p = (struct thread_args *)args;
  for (int i = 0; i < p->count; i++) {
    pthread_mutex_lock(&mutex);
    total_count++;
    pthread_mutex_unlock(&mutex);
  }

  return NULL;
}

void *accumulate_atomic(void *args) {
  struct thread_args *p = (struct thread_args *)args;
  for (int i = 0; i < p->count; i++) {
    __sync_fetch_and_add(&total_count, 1);
  }

  return NULL;
}

int main() {
  pthread_t thread1, thread2;

  struct thread_args args1 = {10000};
  struct thread_args args2 = {15000};

  pthread_create(&thread1, NULL, accumulate_atomic, &args1);
  pthread_create(&thread2, NULL, accumulate_atomic, &args2);

  pthread_join(thread1, NULL);
  pthread_join(thread2, NULL);

  printf("Total count is %d\n", total_count);

  return 0;
}
