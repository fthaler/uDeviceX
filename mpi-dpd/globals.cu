#include <cassert>

#include "common.h"
#include "globals.h"

Globals::Globals() : localcomm(this) {
    // global variables from main.cu
    currently_profiling = false;
}

void Globals::mpiDependentInit(MPI_Comm activecomm) {
    // global variables from main.cu
    localcomm.initialize(activecomm);

    // static variable of class Particle
    MPI_CHECK(MPI_Type_contiguous(6, MPI_FLOAT, &particle_datatype));
    MPI_CHECK(MPI_Type_commit(&particle_datatype));

    // static variable of class Particle
    MPI_CHECK(MPI_Type_contiguous(3, MPI_FLOAT, &acceleration_datatype));
    MPI_CHECK(MPI_Type_commit(&acceleration_datatype));
}
