/*
 *  rbc-interactions.h
 *  Part of CTC/mpi-dpd/
 *
 *  Created and authored by Diego Rossinelli on 2014-12-02.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

#pragma once

#include <vector>
#include <rbc-cuda.h>

#include "common.h"

#include <../dpd-rng.h>

class ComputeFSI
{
    enum { TAGBASE_C = 113, TAGBASE_P = 365, TAGBASE_A = 668, TAGBASE_P2 = 1055, TAGBASE_A2 = 1501 };

protected:

    MPI_Comm cartcomm;

    bool firstpost;

    int nranks, dstranks[26],
	dims[3], periods[3], coords[3], myrank,
	recv_tags[26], recv_counts[26], send_counts[26];

    cudaEvent_t evPpacked, evAcomputed;

    SimpleDeviceBuffer<Particle> packbuf;
    PinnedHostBuffer<Particle> host_packbuf;
    PinnedHostBuffer<int> requiredpacksizes, packstarts_padded;

    std::vector<MPI_Request> reqsendC, reqrecvC, reqsendP, reqrecvP, reqsendA, reqrecvA;

    Logistic::KISS local_trunk;

    struct RemoteHalo
    {
	int expected, capacity;

	SimpleDeviceBuffer<Particle> dstate;
	PinnedHostBuffer<Particle> hstate;
	PinnedHostBuffer<Acceleration> result;

	void preserve_resize(int n)
	    {
		dstate.resize(n);
		hstate.preserve_resize(n);
		result.resize(n);
		capacity = dstate.capacity;

		assert(hstate.capacity == capacity);
	    }

    } remote[26];

    struct LocalHalo
    {
	int expected, capacity;

	SimpleDeviceBuffer<int> scattered_indices;
	PinnedHostBuffer<Acceleration> result;

	void resize(int n) { scattered_indices.resize(n); result.resize(n); capacity = scattered_indices.capacity; }

    } local[26];

    void _wait(std::vector<MPI_Request>& v)
    {
	MPI_Status statuses[v.size()];

	if (v.size())
	    MPI_CHECK(MPI_Waitall(v.size(), &v.front(), statuses));

	v.clear();
    }

    void _postrecvs()
    {
	for(int i = 0; i < 26; ++i)
	{
	    MPI_Request reqC, reqP, reqA;

	    MPI_CHECK( MPI_Irecv(recv_counts + i, 1, MPI_INTEGER, dstranks[i],
				 TAGBASE_C + recv_tags[i], cartcomm,  &reqC) );

	    assert(remote[i].hstate.data);

	    MPI_CHECK( MPI_Irecv(remote[i].hstate.data, remote[i].expected * 6, MPI_FLOAT, dstranks[i],
				 TAGBASE_P + recv_tags[i], cartcomm, &reqP) );

	    assert(local[i].result.data);

	    MPI_CHECK( MPI_Irecv(local[i].result.data, local[i].expected * 3, MPI_FLOAT, dstranks[i],
				 TAGBASE_A + recv_tags[i], cartcomm, &reqA) );

	    reqrecvC.push_back(reqC);
	    reqrecvP.push_back(reqP);
	    reqrecvA.push_back(reqA);
	}
    }

public:

    ComputeFSI(MPI_Comm cartcomm);

    void pack_p(const Particle * const solute, const int nsolute, cudaStream_t stream);

    void post_p(const Particle * const solute, const int nsolute, cudaStream_t stream, cudaStream_t downloadstream);

    void bulk(const Particle * const solvent, const int nsolvent, Acceleration * accsolvent,
	      const int * const cellsstart_solvent, const int * const cellscount_solvent,
	      const Particle * const solute, const int nrbcs, Acceleration * accsolute, cudaStream_t stream);

    void halo(const Particle * const solvent, const int nsolvent, Acceleration * accsolvent,
	      const int * const cellsstart_solvent, const int * const cellscount_solvent,
	      cudaStream_t stream, cudaStream_t uploadstream);

    void post_a();

    void merge_a(Acceleration * accsolute, const int nsolute, cudaStream_t stream);

    ~ComputeFSI();
};