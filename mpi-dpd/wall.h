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
#include "migration.h"
#include "migratable-datastructures.h"

namespace SolidWallsKernel
{
    __global__ void fill_keys(cudaTextureObject_t texSDF, const Particle * const particles, const int n, int * const key);
}

class ComputeWall : public GlobalsInjector, public Migratable<2, 0, 1>
{
    MPI_Comm cartcomm;
    int myrank, dims[3], periods[3], coords[3];

    Logistic::KISS trunk;

    int solid_size;
    float4 * solid4;

    cudaArray_t arrSDF;

    MigratableCellLists cells;

    bool active;

public:
    // global texture references moved from wall.cu
    cudaTextureObject_t texSDF;

    ComputeWall(Globals* globals, MPI_Comm cartcomm);

    void init(Particle* const p, const int n, int& nsurvived, ExpectedMessageSizes& new_sizes, const bool verbose);

    ~ComputeWall();

    void bounce(Particle * const p, const int n, cudaStream_t stream);

    void interactions(const Particle * const p, const int n, Acceleration * const acc,
		      const int * const cellsstart, const int * const cellscount, cudaStream_t stream);

    bool is_active() const { return active; }
    void create_sdf_texture();
    void destroy_sdf_texture();
};
