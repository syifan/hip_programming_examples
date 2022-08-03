// This program calculates PI using Monte Carlo method. Users can select either
// the parallel or serial version.
//
// Arguments:
// - argv[1] - N, number of points to be used in Monte Carlo method.
// - argv[2] - serial or parallel, serial or parallel version of Monte Carlo
// method.
#include <locale.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void print_usage(int argc, char *argv[]) {
  printf("Usage: %s <N> <serial|parallel>\n", argv[0]);
  printf("N - number of points to be used in Monte Carlo method.\n");
  printf("serial - serial version of Monte Carlo method.\n");
  printf("parallel - parallel version of Monte Carlo method.\n");
}

uint64_t current_time_in_ns() {
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  return ts.tv_sec * 1000000000 + ts.tv_nsec;
}

double monte_carlo_pi_serial(int n) {
  double x, y;
  double pi = 0;

  for (int i = 0; i < n; i++) {
    // Generate random numbers between -1 and 1.
    x = (double)rand() / RAND_MAX * 2 - 1;
    y = (double)rand() / RAND_MAX * 2 - 1;
    if (x * x + y * y <= 1) {
      pi++;
    }
  }

  return 4 * pi / n;
}

struct monte_carlo_pi_args {
  int n;
  double pi;
};

void *monte_carlo_pi_thread(void *data) {
  struct monte_carlo_pi_args *args = (struct monte_carlo_pi_args *)data;
  args->pi = monte_carlo_pi_serial(args->n);
  return NULL;
}

double monte_carlo_pi_parallel(int n) {
  int n_threads = 12;

  pthread_t threads[n_threads];
  struct monte_carlo_pi_args args[n_threads];

  // Create threads.
  for (int i = 0; i < n_threads; i++) {
    args[i].n = n / n_threads;
    pthread_create(&threads[i], NULL, monte_carlo_pi_thread, &args[i]);
  }

  // Wait for threads to finish.
  for (int i = 0; i < n_threads; i++) {
    pthread_join(threads[i], NULL);
  }

  // Calculate PI.
  double pi = 0;
  for (int i = 0; i < n_threads; i++) {
    pi += args[i].pi;
  }
  pi /= n_threads;

  return pi;
}

int main(int argc, char *argv[]) {
  // Print usage if invalid number of arguments.
  if (argc != 3) {
    print_usage(argc, argv);
    return 1;
  }

  // Parse arguments.
  int n = atoi(argv[1]);
  char *version = argv[2];

  // Check if arguments are valid.
  if (n <= 0) {
    printf("N is not valid.\n");
    print_usage(argc, argv);
    return 1;
  }

  if (strcmp(version, "serial") != 0 && strcmp(version, "parallel") != 0) {
    printf("Version is not valid.\n");
    print_usage(argc, argv);
    return 1;
  }

  uint64_t start_time, end_time;
  double pi = 0;
  if (strcmp(version, "serial") == 0) {
    start_time = current_time_in_ns();
    pi = monte_carlo_pi_serial(n);
    end_time = current_time_in_ns();
  } else if (strcmp(version, "parallel") == 0) {
    start_time = current_time_in_ns();
    pi = monte_carlo_pi_parallel(n);
    end_time = current_time_in_ns();
  }

  setlocale(LC_NUMERIC, "");
  printf("PI = %.10f\n", pi);
  printf("Time = %'llu ns\n", end_time - start_time);

  return 0;
}