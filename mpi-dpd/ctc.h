/*
 *  ctc.h
 *  Part of uDeviceX/mpi-dpd/
 *
 *  Created and authored by Diego Rossinelli on 2014-12-18.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

#pragma once

#include "redistribute-rbcs.h"
#include "minmax.h"

#include <ctc-cuda.h>

class RedistributeCTCs : public RedistributeRBCs
{
    void _compute_extents(const Particle * const xyzuvw, const int nrbcs, cudaStream_t stream)
    {
	assert(sizeof(CudaCTC::Extent) == sizeof(CudaRBC::Extent));
#if 1
	if (nrbcs)
	    minmax(xyzuvw, nvertices, nrbcs, minextents.devptr, maxextents.devptr, stream);
#else
	for(int i = 0; i < nrbcs; ++i)
	    CudaCTC::extent_nohost(stream, (float *)(xyzuvw + nvertices * i), (CudaCTC::Extent *)(extents.devptr + i));
#endif
    }

public:

RedistributeCTCs(Globals* globals, MPI_Comm _cartcomm):RedistributeRBCs(globals, _cartcomm)
    {
	if (globals->ctcs)
	{
	    CudaCTC::Extent host_extent;
        CudaCTC::setup(nvertices, host_extent);
	}
    }
};

class CollectionCTC : public CollectionBase
{
protected:
    void _initialize(float *device_xyzuvw, const float (*transform)[4])
    {
	    CudaCTC::initialize(device_xyzuvw, transform);
    }

public:
    CollectionCTC(Globals* globals, MPI_Comm cartcomm)
        : CollectionBase(globals, cartcomm)
    {
	    CudaCTC::Extent extent;
	    CudaCTC::setup(nvertices, extent);

	    assert(extent.xmax - extent.xmin < XSIZE_SUBDOMAIN);
	    assert(extent.ymax - extent.ymin < YSIZE_SUBDOMAIN);
	    assert(extent.zmax - extent.zmin < ZSIZE_SUBDOMAIN);

	    CudaCTC::get_triangle_indexing(indices, ntriangles);
    }

    void dump(MPI_Comm comm, MPI_Comm cartcomm, Particle * const p, const Acceleration * const a, const int n, const int iddatadump)
    {
	    _dump(globals, "xyz/ctcs.xyz", "ply/ctcs-%04d.ply", comm, cartcomm, ntriangles, n / nvertices, nvertices, indices, p, a, n, iddatadump);
    }
};
