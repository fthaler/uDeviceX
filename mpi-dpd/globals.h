#pragma once

#include <mpi.h>

class Globals {
public:
    Globals();

    // class Particle
    MPI_Datatype particle_datatype;
    
    // class Acceleration
    MPI_Datatype acceleration_datatype;

    // class CollectionRBC
    int (*collectionrbc_indices)[3];
    int collectionrbc_ntriangles;
    int collectionrbc_nvertices;

    // class CollectionCTC
    int (*collectionctc_indices)[3];
    int collectionctc_ntriangles;
    int collectionctc_nvertices;
};
