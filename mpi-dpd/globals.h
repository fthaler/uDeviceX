#pragma once

#include <mpi.h>

class Globals {
public:
    Globals();

    // static variable of class Particle
    MPI_Datatype particle_datatype;
    // static variable of class Particle
    MPI_Datatype acceleration_datatype;
};
