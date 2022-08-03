// This program calculates PI using Monte Carlo method. Users can select either
// the parallel or serial version.
//
// Arguments:
// - argv[1] - N, number of points to be used in Monte Carlo method.
// - argv[2] - serial or parallel, serial or parallel version of Monte Carlo
// method.
// - argv[3] - number of threads, number of threads to be used in parallel
// version of Monte Carlo method.

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
  printf(
      "number of threads - number of threads to be used in parallel version of "
      "Monte Carlo method.\n");
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
  int num_threads;
};

void *monte_carlo_pi_thread(void *data) {
  struct monte_carlo_pi_args *args = (struct monte_carlo_pi_args *)data;

  printf("Thread started, n=%d.\n", args->n);

  args->pi = monte_carlo_pi_serial(args->n);
  return NULL;
}

double monte_carlo_pi_parallel(int n, int num_threads) {
  printf("Calculating PI using %d threads...\n", num_threads);

  pthread_t threads[num_threads];
  struct monte_carlo_pi_args args[num_threads];

  // Create threads.
  for (int i = 0; i < num_threads; i++) {
    args[i].n = n / num_threads;
    pthread_create(&threads[i], NULL, monte_carlo_pi_thread, &args[i]);
  }

  // Wait for threads to finish.
  for (int i = 0; i < num_threads; i++) {
    pthread_join(threads[i], NULL);
  }

  // Calculate PI.
  double pi = 0;
  for (int i = 0; i < num_threads; i++) {
    pi += args[i].pi;
  }
  pi /= num_threads;

  return pi;
}

struct Args {
  int n;
  char *version;
  int num_threads;
};

struct Args parse_args(int argc, char *argv[]) {
  struct Args args;

  // Print usage if invalid number of arguments.
  if (argc < 3) {
    print_usage(argc, argv);
    exit(1);
  }

  // Parse arguments.
  args.n = atoi(argv[1]);
  args.version = argv[2];

  // Check if arguments are valid.
  if (args.n <= 0) {
    printf("N is not valid.\n");
    print_usage(argc, argv);
    exit(1);
  }

  if (strcmp(args.version, "serial") != 0 &&
      strcmp(args.version, "parallel") != 0) {
    printf("Version is not valid.\n");
    print_usage(argc, argv);
    exit(1);
  }

  if (strcmp(args.version, "serial") == 0) {
    return args;
  }

  if (strcmp(args.version, "parallel") == 0 && argc < 4) {
    printf("Number of threads is not valid.\n");
    print_usage(argc, argv);
    exit(1);
  }

  args.num_threads = atoi(argv[3]);
  if (args.num_threads <= 0) {
    printf("Number of threads is not valid.\n");
    print_usage(argc, argv);
    exit(1);
  }

  return args;
}

int main(int argc, char *argv[]) {
  setlocale(LC_NUMERIC, "");

  struct Args args = parse_args(argc, argv);

  uint64_t start_time, end_time;
  double pi = 0;

  if (strcmp(args.version, "serial") == 0) {
    start_time = current_time_in_ns();
    pi = monte_carlo_pi_serial(args.n);
    end_time = current_time_in_ns();
  } else if (strcmp(args.version, "parallel") == 0) {
    start_time = current_time_in_ns();
    pi = monte_carlo_pi_parallel(args.n, args.num_threads);
    end_time = current_time_in_ns();
  }

  printf("PI = %.10f\n", pi);
  printf("Time = %'llu ns\n", end_time - start_time);

  return 0;
}
