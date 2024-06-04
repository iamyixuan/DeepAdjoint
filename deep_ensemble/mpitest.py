from mpi4py import MPI

def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD

    # Get the size of the MPI world and the rank of this process
    size = comm.Get_size()
    rank = comm.Get_rank()

    # Print a message from each process
    print(f"Hello from process {rank} out of {size}")

    # Finalize MPI (not strictly necessary in a script that is about to exit)
    MPI.Finalize()

if __name__ == "__main__":
    main()