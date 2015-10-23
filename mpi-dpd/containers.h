/*
 *  containers.h
 *  Part of uDeviceX/mpi-dpd/
 *
 *  Created and authored by Diego Rossinelli on 2014-12-05.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

#pragma once

#include <vector>
#include <string>

#include "common.h"
#include "globals.h"
#include "migratable-datastructures.h"

struct ParticleArray : public GlobalsInjector
{
    int size;

    float3 origin, globalextent;

    MigratableDeviceBuffer<Particle> xyzuvw;
    MigratableDeviceBuffer<Acceleration> axayaz;

    ParticleArray(Globals* globals = NULL) : GlobalsInjector(globals) {}

    virtual void resize(int n);
    virtual void preserve_resize(int n);
    void update_stage1(const float driving_acceleration, cudaStream_t stream);
    void update_stage2_and_1(const float driving_acceleration, cudaStream_t stream);
    void clear_velocity();

    void clear_acc(cudaStream_t stream)
	{
	    CUDA_CHECK(cudaMemsetAsync(axayaz.data, 0, sizeof(Acceleration) * axayaz.size, stream));
	}
};

class CollectionBase : public ParticleArray
{
protected:
    int (*indices)[3];
    int ntriangles;
    int nvertices;

    MPI_Comm cartcomm;

    int ncells, myrank, dims[3], periods[3], coords[3];

    virtual void _initialize(float *device_xyzuvw, const float (*transform)[4]) = 0;

    static void _dump(Globals* globals, const char * const path2xyz, const char * const format4ply,
		      MPI_Comm comm, MPI_Comm cartcomm, const int ntriangles, const int ncells, const int nvertices,
		      int (* const indices)[3],
		      Particle * const p, const Acceleration * const a, const int n, const int iddatadump);

public:
    int get_nvertices() const { return nvertices; }

    CollectionBase(Globals* globals, MPI_Comm cartcomm);

    void setup(const char* const path2ic);

    Particle* data() { return xyzuvw.data; }
    Acceleration* acc() { return axayaz.data; }

    void remove(const int* const entries, const int nentries);
    virtual void resize(const int rbcs_count);
    virtual void preserve_resize(int n);

    int count() { return ncells; }
    int pcount() { return ncells * nvertices; }
    virtual void dump(MPI_Comm comm, MPI_Comm cartcomm,
		     Particle * const p, const Acceleration * const a, const int n, const int iddatadump) = 0;
};

class CollectionRBC : public CollectionBase
{
protected:
    void _initialize(float *device_xyzuvw, const float (*transform)[4]);
public:
    CollectionRBC(Globals* globals, MPI_Comm cartcomm);
    void dump(MPI_Comm comm, MPI_Comm cartcomm,
		     Particle * const p, const Acceleration * const a, const int n, const int iddatadump)
    {
	    _dump(globals, "xyz/rbcs.xyz", "ply/rbcs-%04d.ply", comm, cartcomm,
              ntriangles, n / nvertices, nvertices, indices, p, a, n, iddatadump);
    }
};
