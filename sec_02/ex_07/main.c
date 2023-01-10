#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

uint64_t current_time_in_ns() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000000000 + ts.tv_nsec;
}

int main(int argc, char** argv) {
  // Initialize the MPI environment
  MPI_Init(&argc, &argv);

  // Get the number of processes
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // Get the rank of the process
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  // Get the name of the processor
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);

  int start_number, end_number = 0;
  unsigned int seed = current_time_in_ns();

  // Rank 0 generate a random start number
  if (world_rank == 0) {
    start_number = rand_r(&seed) % 100;
  }

  // Broadcast the start number to all processes
  MPI_Bcast(&start_number, 1, MPI_INT, 0, MPI_COMM_WORLD);
  printf("Process %d received start number %d\n", world_rank, start_number);

  // Add a random number to the start number
  int random_number = rand_r(&seed) % 100;
  start_number += random_number;
  printf("Process %d added a random number %d and now has number %d\n",
         world_rank, random_number, start_number);

  // Barrier to make sure all processes have reached this point
  MPI_Barrier(MPI_COMM_WORLD);

  // Average the start number over all processes
  MPI_Reduce(&start_number, &end_number, 1, MPI_INT, MPI_SUM, 0,
             MPI_COMM_WORLD);

  // Print the average end number
  if (world_rank == 0) {
    printf("The average number is %d\n", end_number / world_size);
  }

  // Finalize the MPI environment.
  MPI_Finalize();
}