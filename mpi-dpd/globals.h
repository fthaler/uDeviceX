#pragma once

#include <mpi.h>

#include "common.h"

class Globals {
public:
    Globals();

    // function for initializing stuff after MPI_Init
    void mpiDependentInit(MPI_Comm activecomm);

    // global variables from main.cu
    bool currently_profiling;
    float tend;
    bool walls, pushtheflow, doublepoiseuille, rbcs, ctcs;
    bool xyz_dumps, hdf5field_dumps, hdf5part_dumps;
    bool is_mps_enabled, adjust_message_sizes, contactforces;
    int steps_per_report, steps_per_dump, wall_creation_stepid;
    int nvtxstart, nvtxstop;
    LocalComm localcomm;

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

    // globals from fsi.cu
    cudaTextureObject_t fsi_texSolventParticles;
    cudaTextureObject_t fsi_texCellsStart;
    /* currently unused
    cudaTextureObject_t fsi_texCellsCount;*/
    bool fsi_firsttime;
};

class GlobalsInjector
{
public:
    Globals* globals;

    GlobalsInjector(Globals* globals) : globals(globals) {}
};
