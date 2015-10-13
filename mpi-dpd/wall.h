/*
 *  wall.h
 *  Part of uDeviceX/mpi-dpd/
 *
 *  Created and authored by Diego Rossinelli on 2014-11-19.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

#pragma once

#include <mpi.h>

#include <../dpd-rng.h>
#include "common.h"
#include "globals.h"

namespace SolidWallsKernel
{
    __global__ void fill_keys(cudaTextureObject_t texSDF, const Particle * const particles, const int n, int * const key);
}

class ComputeWall
{
    // global variable pointer for AMPI
    Globals* globals;

    MPI_Comm cartcomm;
    int myrank, dims[3], periods[3], coords[3];

    Logistic::KISS trunk;

    int solid_size;
    float4 * solid4;

    cudaArray * arrSDF;

    CellLists cells;

public:
    // global texture references moved from wall.cu
    cudaTextureObject_t texSDF;
    cudaTextureObject_t texWallParticles;
    cudaTextureObject_t texWallCellStart;
    /* currently unused
    texWallCellCount; */

    ComputeWall(Globals* globals, MPI_Comm cartcomm, Particle* const p, const int n, int& nsurvived, ExpectedMessageSizes& new_sizes, const bool verbose);

    ~ComputeWall();

    void bounce(Particle * const p, const int n, cudaStream_t stream);

    void interactions(const Particle * const p, const int n, Acceleration * const acc,
		      const int * const cellsstart, const int * const cellscount, cudaStream_t stream);
};
