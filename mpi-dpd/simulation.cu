/*
 *  simulation.cu
 *  Part of uDeviceX/mpi-dpd/
 *
 *  Created and authored by Diego Rossinelli on 2015-03-24.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

#include <sys/stat.h>

#include "simulation.h"

__global__ void make_texture( float4 * __restrict xyzouvwo, ushort4 * __restrict xyzo_half, const float * __restrict xyzuvw, const uint n )
{
    extern __shared__ volatile float  smem[];
    const uint warpid = threadIdx.x / 32;
    const uint lane = threadIdx.x % 32;

    const uint i =  (blockIdx.x * blockDim.x + threadIdx.x ) & 0xFFFFFFE0U;

    const float2 * base = ( float2* )( xyzuvw +  i * 6 );
#pragma unroll 3
    for( uint j = lane; j < 96; j += 32 ) {
	float2 u = base[j];
	// NVCC bug: no operator = between volatile float2 and float2
	asm volatile( "st.volatile.shared.v2.f32 [%0], {%1, %2};" : : "r"( ( warpid * 96 + j )*8 ), "f"( u.x ), "f"( u.y ) : "memory" );
    }
    // SMEM: XYZUVW XYZUVW ...
    uint pid = lane / 2;
    const uint x_or_v = ( lane % 2 ) * 3;
    xyzouvwo[ i * 2 + lane ] = make_float4( smem[ warpid * 192 + pid * 6 + x_or_v + 0 ],
					    smem[ warpid * 192 + pid * 6 + x_or_v + 1 ],
					    smem[ warpid * 192 + pid * 6 + x_or_v + 2 ], 0 );
    pid += 16;
    xyzouvwo[ i * 2 + lane + 32] = make_float4( smem[ warpid * 192 + pid * 6 + x_or_v + 0 ],
						smem[ warpid * 192 + pid * 6 + x_or_v + 1 ],
						smem[ warpid * 192 + pid * 6 + x_or_v + 2 ], 0 );

    xyzo_half[i + lane] = make_ushort4( __float2half_rn( smem[ warpid * 192 + lane * 6 + 0 ] ),
					__float2half_rn( smem[ warpid * 192 + lane * 6 + 1 ] ),
					__float2half_rn( smem[ warpid * 192 + lane * 6 + 2 ] ), 0 );
// }
}

void Simulation::_update_helper_arrays()
{
    CUDA_CHECK( cudaFuncSetCacheConfig( make_texture, cudaFuncCachePreferShared ) );

    const int np = particles->size;

    xyzouvwo.resize(2 * np);
    xyzo_half.resize(np);

    if (np) {
    make_texture <<< (np + 1023) / 1024, 1024, 1024 * 6 * sizeof( float )>>>(xyzouvwo.data, xyzo_half.data, (float *)particles->xyzuvw.data, np );
    AMPI_YIELD(activecomm);
    }

    CUDA_CHECK(cudaPeekAtLastError());
}

std::vector<Particle> Simulation::_ic()
{
    srand48(rank);

    std::vector<Particle> ic(XSIZE_SUBDOMAIN * YSIZE_SUBDOMAIN * ZSIZE_SUBDOMAIN * numberdensity);

    const int L[3] = { XSIZE_SUBDOMAIN, YSIZE_SUBDOMAIN, ZSIZE_SUBDOMAIN };

    for(int iz = 0; iz < L[2]; iz++)
	for(int iy = 0; iy < L[1]; iy++)
	    for(int ix = 0; ix < L[0]; ix++)
		for(int l = 0; l < numberdensity; ++l)
		{
		    const int p = l + numberdensity * (ix + L[0] * (iy + L[1] * iz));

		    ic[p].x[0] = -L[0]/2 + ix + 0.99 * drand48();
		    ic[p].x[1] = -L[1]/2 + iy + 0.99 * drand48();
		    ic[p].x[2] = -L[2]/2 + iz + 0.99 * drand48();
		    ic[p].u[0] = 0;
		    ic[p].u[1] = 0;
		    ic[p].u[2] = 0;
		}

    /* use this to check robustness
       for(int i = 0; i < ic.size(); ++i)
       for(int c = 0; c < 3; ++c)
       {
       ic[i].x[c] = -L[c] * 0.5 + drand48() * L[c];
       ic[i].u[c] = 0;
       }
    */

    return ic;
}

void Simulation::_redistribute()
{
    double tstart = MPI_Wtime();

    redistribute.pack(particles->xyzuvw.data, particles->size, mainstream);

    CUDA_CHECK(cudaPeekAtLastError());

    if (rbcscoll)
	redistribute_rbcs.extent(rbcscoll->data(), rbcscoll->count(), mainstream);

    if (ctcscoll)
	redistribute_ctcs.extent(ctcscoll->data(), ctcscoll->count(), mainstream);

    redistribute.send();

    if (rbcscoll)
	redistribute_rbcs.pack_sendcount(rbcscoll->data(), rbcscoll->count(), mainstream);

    if (ctcscoll)
	redistribute_ctcs.pack_sendcount(ctcscoll->data(), ctcscoll->count(), mainstream);

    redistribute.bulk(particles->size, cells.start, cells.count, mainstream);

    CUDA_CHECK(cudaPeekAtLastError());

    const int newnp = redistribute.recv_count(mainstream, host_idle_time);

    int nrbcs;
    if (rbcscoll)
	nrbcs = redistribute_rbcs.post();

    int nctcs;
    if (ctcscoll)
	nctcs = redistribute_ctcs.post();

    if (rbcscoll)
	rbcscoll->resize(nrbcs);

    if (ctcscoll)
	ctcscoll->resize(nctcs);

    newparticles->resize(newnp);
    xyzouvwo.resize(newnp * 2);
    xyzo_half.resize(newnp);

    redistribute.recv_unpack(newparticles->xyzuvw.data, xyzouvwo.data, xyzo_half.data, newnp, cells.start, cells.count, mainstream, host_idle_time);

    CUDA_CHECK(cudaPeekAtLastError());

    swap(particles, newparticles);

    if (rbcscoll)
	redistribute_rbcs.unpack(rbcscoll->data(), rbcscoll->count(), mainstream);

    if (ctcscoll)
	redistribute_ctcs.unpack(ctcscoll->data(), ctcscoll->count(), mainstream);

    CUDA_CHECK(cudaPeekAtLastError());

    //globals->localcomm.barrier();

    timings["redistribute"] += MPI_Wtime() - tstart;
}

void Simulation::_report(const bool verbose, const int idtimestep)
{
    report_host_memory_usage(activecomm, stdout);

    {
	double t1 = MPI_Wtime();

	float host_busy_time = (MPI_Wtime() - report_t0_a) - host_idle_time;

	host_busy_time *= 1e3 / globals->steps_per_report;

	float sumval, maxval, minval;
	MPI_CHECK(MPI_Reduce(&host_busy_time, &sumval, 1, MPI_FLOAT, MPI_SUM, 0, activecomm));
	MPI_CHECK(MPI_Reduce(&host_busy_time, &maxval, 1, MPI_FLOAT, MPI_MAX, 0, activecomm));
	MPI_CHECK(MPI_Reduce(&host_busy_time, &minval, 1, MPI_FLOAT, MPI_MIN, 0, activecomm));

	int commsize;
	MPI_CHECK(MPI_Comm_size(activecomm, &commsize));

#ifdef AMPI
    // use static variables for intra-PE communication
    static volatile float pe_busy_time, vp_busy_time_min, vp_busy_time_max;
    static volatile int vps_per_pe;
    pe_busy_time = 0;
    vp_busy_time_min = host_busy_time;
    vp_busy_time_max = host_busy_time;
    vps_per_pe = 0;
    MPI_CHECK(MPI_Barrier(activecomm));
    // get data of all PEs
    pe_busy_time += host_busy_time;
    if (host_busy_time > vp_busy_time_max)
        vp_busy_time_max = host_busy_time;
    if (host_busy_time < vp_busy_time_min)
        vp_busy_time_min = host_busy_time;
    const int vpid = vps_per_pe++;
    MPI_CHECK(MPI_Barrier(activecomm));

    if (idtimestep > 0) {
        // report info per PE
        if (vpid == 0) {
            printf("\x1b[95mPE %d workload (%d VPs) min/avg/max/total: %.2f/%.2f/%.2f ms\x1b[0m\n",
                   MPI_My_pe(), vps_per_pe, vp_busy_time_min, pe_busy_time / vps_per_pe, vp_busy_time_max, pe_busy_time);
        }

        // get and report PE-imbalance
        float pe_busy_time_s = vpid == 0 ? pe_busy_time : 0.0f;
        float pe_busy_time_sum, pe_busy_time_max;
        MPI_CHECK(MPI_Reduce(&pe_busy_time_s, &pe_busy_time_sum, 1, MPI_FLOAT, MPI_SUM, 0, activecomm));
        MPI_CHECK(MPI_Reduce(&pe_busy_time_s, &pe_busy_time_max, 1, MPI_FLOAT, MPI_MAX, 0, activecomm));
        const double pe_imbalance = 100 * (pe_busy_time_max / pe_busy_time_sum * MPI_Num_pes() - 1);
        if (verbose)
            printf("\x1b[95moverall PE imbalance: %.f%%\n", pe_imbalance);

        // compute load as a mix of average PE load and VP load
        const float vp_mix = 0.1f;
        const float load = (pe_busy_time / vps_per_pe) * (1.0f - vp_mix) + host_busy_time * vp_mix;
        //MPI_Set_load(load);
    }
#endif

	const double imbalance = 100 * (maxval / sumval * commsize - 1);

	if (verbose && imbalance >= 0)
	    printf("\x1b[93moverall imbalance: %.f%%, host workload min/avg/max: %.2f/%.2f/%.2f ms\x1b[0m\n",
		   imbalance , minval, sumval / commsize, maxval);

	globals->localcomm.print_particles(particles->size);

	host_idle_time = 0;
	report_t0_a = t1;
    }

    {
	double t1 = MPI_Wtime();

	if (verbose)
	{
	    printf("\x1b[92mbeginning of time step %d (%.3f ms)\x1b[0m\n", idtimestep, (t1 - report_t0_b) * 1e3 / globals->steps_per_report);
	    printf("in more details, per time step:\n");
	    double tt = 0;
	    for(TimingsMap::iterator it = timings.begin(); it != timings.end(); ++it)
	    {
		printf("%s: %.3f ms\n", it->first, it->second * 1e3 / globals->steps_per_report);
		tt += it->second;
		it->second = 0;
	    }
	    printf("discrepancy: %.3f ms\n", ((t1 - report_t0_b) - tt) * 1e3 / globals->steps_per_report);
	}

	report_t0_b = t1;
    }
}

void Simulation::_remove_bodies_from_wall(CollectionBase * coll)
{
    if (!coll || !coll->count())
	return;

    SimpleDeviceBuffer<int> marks(coll->pcount());

    assert(wall.is_active());
    SolidWallsKernel::fill_keys<<< (coll->pcount() + 127) / 128, 128 >>>(wall.texSDF, coll->data(), coll->pcount(), marks.data);
    AMPI_YIELD(activecomm);

    vector<int> tmp(marks.size);
    CUDA_CHECK(cudaMemcpy(tmp.data(), marks.data, sizeof(int) * marks.size, cudaMemcpyDeviceToHost));

    const int nbodies = coll->count();
    const int nvertices = coll->get_nvertices();

    std::vector<int> tokill;
    for(int i = 0; i < nbodies; ++i)
    {
	bool valid = true;

	for(int j = 0; j < nvertices && valid; ++j)
	    valid &= 0 == tmp[j + nvertices * i];

	if (!valid)
	    tokill.push_back(i);
    }

    coll->remove(&tokill.front(), tokill.size());
    coll->clear_velocity();

    CUDA_CHECK(cudaPeekAtLastError());
}

void Simulation::_create_walls(const bool verbose, bool & termination_request)
{
    if (verbose)
	printf("creation of the walls...\n");

    int nsurvived = 0;
    ExpectedMessageSizes new_sizes;
    wall.init(particles->xyzuvw.data, particles->size, nsurvived, new_sizes, verbose);

    //adjust the message sizes if we're pushing the flow in x
    {
	const double xvelavg = getenv("XVELAVG") ? atof(getenv("XVELAVG")) : globals->pushtheflow;
	const double yvelavg = getenv("YVELAVG") ? atof(getenv("YVELAVG")) : 0;
	const double zvelavg = getenv("ZVELAVG") ? atof(getenv("ZVELAVG")) : 0;

	for(int code = 0; code < 27; ++code)
	{
	    const int d[3] = {
		(code % 3) - 1,
		((code / 3) % 3) - 1,
		((code / 9) % 3) - 1
	    };

	    const double IudotnI =
		fabs(d[0] * xvelavg) +
		fabs(d[1] * yvelavg) +
		fabs(d[2] * zvelavg) ;

	    const float factor = 1 + IudotnI * dt * 10 * numberdensity;

	    //printf("RANK %d: direction %d %d %d -> IudotnI is %f and final factor is %f\n",
	    //rank, d[0], d[1], d[2], IudotnI, 1 + IudotnI * dt * numberdensity);

	    new_sizes.msgsizes[code] *= factor;
	}
    }

    //MPI_CHECK(MPI_Barrier(activecomm));
    //redistribute.adjust_message_sizes(new_sizes);
    //dpd->adjust_message_sizes(new_sizes);
    //MPI_CHECK(MPI_Barrier(activecomm));

    //there is no support for killing zero-workload ranks for rbcs and ctcs just yet
    /* this is unnecessarily complex for now
       if (!globals->rbcs && !globals->ctcs)
       {
       const bool local_work = new_sizes.msgsizes[1 + 3 + 9] > 0;

       MPI_CHECK(MPI_Comm_split(cartcomm, local_work, rank, &activecomm)) ;

       MPI_CHECK(MPI_Comm_rank(activecomm, &rank));

       if (!local_work )
       {
       if (rank == 0)
       {
       int nkilled;
       MPI_CHECK(MPI_Comm_size(activecomm, &nkilled));

       printf("THERE ARE %d RANKS WITH ZERO WORKLOAD THAT WILL MPI-FINALIZE NOW.\n", nkilled);
       }

       termination_request = true;
       return;
       }
       }
    */

    particles->resize(nsurvived);
    particles->clear_velocity();
    cells.build(particles->xyzuvw.data, particles->size, 0, NULL, NULL);

    _update_helper_arrays();

    CUDA_CHECK(cudaPeekAtLastError());

    //remove cells touching the wall
    _remove_bodies_from_wall(rbcscoll);
    _remove_bodies_from_wall(ctcscoll);

    {
	H5PartDump sd("survived-particles->h5part", activecomm, cartcomm);
	Particle * p = new Particle[particles->size];

	CUDA_CHECK(cudaMemcpy(p, particles->xyzuvw.data, sizeof(Particle) * particles->size, cudaMemcpyDeviceToHost));

	sd.dump(p, particles->size);

	delete [] p;
    }
}

void Simulation::_forces()
{
    double tstart = MPI_Wtime();

    SolventWrap wsolvent(particles->xyzuvw.data, particles->size, particles->axayaz.data, cells.start, cells.count);

    std::vector<ParticlesWrap> wsolutes;

    if (rbcscoll)
	wsolutes.push_back(ParticlesWrap(rbcscoll->data(), rbcscoll->pcount(), rbcscoll->acc()));

    if (ctcscoll)
	wsolutes.push_back(ParticlesWrap(ctcscoll->data(), ctcscoll->pcount(), ctcscoll->acc()));

    fsi->bind_solvent(wsolvent);

    solutex->bind_solutes(wsolutes);

    particles->clear_acc(mainstream);

    if (rbcscoll)
	rbcscoll->clear_acc(mainstream);

    if (ctcscoll)
    	ctcscoll->clear_acc(mainstream);

    dpd->pack(particles->xyzuvw.data, particles->size, cells.start, cells.count, mainstream);

    solutex->pack_p(mainstream);

    CUDA_CHECK(cudaPeekAtLastError());

    if (globals->contactforces)
	contact->build_cells(wsolutes, mainstream);

    dpd->local_interactions(particles->xyzuvw.data, xyzouvwo.data, xyzo_half.data, particles->size, particles->axayaz.data,
			   cells.start, cells.count, mainstream);

    dpd->post(particles->xyzuvw.data, particles->size, mainstream, downloadstream);

    solutex->post_p(mainstream, downloadstream);

    CUDA_CHECK(cudaPeekAtLastError());

    if (rbcscoll && wall.is_active())
	wall.interactions(rbcscoll->data(), rbcscoll->pcount(), rbcscoll->acc(), NULL, NULL, mainstream);

    if (ctcscoll && wall.is_active())
	wall.interactions(ctcscoll->data(), ctcscoll->pcount(), ctcscoll->acc(), NULL, NULL, mainstream);

    if (wall.is_active())
	wall.interactions(particles->xyzuvw.data, particles->size, particles->axayaz.data,
			   cells.start, cells.count, mainstream);

    CUDA_CHECK(cudaPeekAtLastError());

    dpd->recv(mainstream, uploadstream);

    solutex->recv_p(uploadstream);

    solutex->halo(uploadstream, mainstream);

    dpd->remote_interactions(particles->xyzuvw.data, particles->size, particles->axayaz.data, mainstream, uploadstream);

    fsi->bulk(wsolutes, mainstream);

    if (globals->contactforces)
	contact->bulk(wsolutes, mainstream);

    CUDA_CHECK(cudaPeekAtLastError());

    if (rbcscoll)
    	CudaRBC::forces_nohost(mainstream, rbcscoll->count(), (float *)rbcscoll->data(), (float *)rbcscoll->acc());

    if (ctcscoll)
	CudaCTC::forces_nohost(mainstream, ctcscoll->count(), (float *)ctcscoll->data(), (float *)ctcscoll->acc());

    CUDA_CHECK(cudaPeekAtLastError());

    solutex->post_a();

    solutex->recv_a(mainstream);

    timings["interactions"] += MPI_Wtime() - tstart;

    CUDA_CHECK(cudaPeekAtLastError());
}

void Simulation::_datadump(const int idtimestep)
{
    double tstart = MPI_Wtime();

#ifdef AMPI
	CUDA_CHECK(cudaEventCreate(&evdownloaded, cudaEventDisableTiming | cudaEventBlockingSync));
#else
    pthread_mutex_lock(&mutex_datadump);

    while (datadump_pending)
	pthread_cond_wait(&done_datadump, &mutex_datadump);
#endif

    int n = particles->size;

    if (rbcscoll)
	n += rbcscoll->pcount();

    if (ctcscoll)
	n += ctcscoll->pcount();

    particles_datadump.resize(n);
    accelerations_datadump.resize(n);

    CUDA_CHECK(cudaMemcpyAsync(particles_datadump.data, particles->xyzuvw.data, sizeof(Particle) * particles->size, cudaMemcpyDeviceToHost,0));
    CUDA_CHECK(cudaMemcpyAsync(accelerations_datadump.data, particles->axayaz.data, sizeof(Acceleration) * particles->size, cudaMemcpyDeviceToHost,0));

    int start = particles->size;

    if (rbcscoll)
    {
	CUDA_CHECK(cudaMemcpyAsync(particles_datadump.data + start, rbcscoll->xyzuvw.data, sizeof(Particle) * rbcscoll->pcount(), cudaMemcpyDeviceToHost, 0));
	CUDA_CHECK(cudaMemcpyAsync(accelerations_datadump.data + start, rbcscoll->axayaz.data, sizeof(Acceleration) * rbcscoll->pcount(), cudaMemcpyDeviceToHost, 0));

	start += rbcscoll->pcount();
    }

    if (ctcscoll)
    {
	CUDA_CHECK(cudaMemcpyAsync(particles_datadump.data + start, ctcscoll->xyzuvw.data, sizeof(Particle) * ctcscoll->pcount(), cudaMemcpyDeviceToHost, 0));
	CUDA_CHECK(cudaMemcpyAsync(accelerations_datadump.data + start, ctcscoll->axayaz.data, sizeof(Acceleration) * ctcscoll->pcount(), cudaMemcpyDeviceToHost, 0));

	start += ctcscoll->pcount();
    }

    assert(start == n);

    CUDA_CHECK(cudaEventRecord(evdownloaded, 0));

#ifdef AMPI
    _datadump_ampi(idtimestep);
#else
    datadump_idtimestep = idtimestep;
    datadump_nsolvent = particles->size;
    datadump_nrbcs = rbcscoll ? rbcscoll->pcount() : 0;
    datadump_nctcs = ctcscoll ? ctcscoll->pcount() : 0;
    datadump_pending = true;

    pthread_cond_signal(&request_datadump);
#if defined(_SYNC_DUMPS_)
    while (datadump_pending)
	pthread_cond_wait(&done_datadump, &mutex_datadump);
#endif

    pthread_mutex_unlock(&mutex_datadump);
#endif // AMPI

    timings["data-dump"] += MPI_Wtime() - tstart;
}

#ifdef AMPI
void Simulation::_datadump_ampi(const int idtimestep)
{
    int rank;
    MPI_CHECK(MPI_Comm_rank(activecomm, &rank));
    bool wallcreated = false;

    if (rank == 0)
	    mkdir("xyz", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

	CUDA_CHECK(cudaEventSynchronize(evdownloaded));

	const int n = particles_datadump.size;
	Particle * p = particles_datadump.data;
	Acceleration * a = accelerations_datadump.data;

	{
	    NVTX_RANGE("diagnostics", NVTX_C1);
	    diagnostics(activecomm, cartcomm, p, n, dt, idtimestep, a);
	}

	if (globals->xyz_dumps)
	{
	    NVTX_RANGE("xyz dump", NVTX_C2);

	    if (globals->walls && idtimestep >= globals->wall_creation_stepid && !wallcreated)
	    {
            if (rank == 0)
            {
                if(access("xyz/particles-equilibration.xyz", F_OK ) == -1)
                   rename("xyz/particles.xyz", "xyz/particles-equilibration.xyz");

                if(access("xyz/rbcs-equilibration.xyz", F_OK ) == -1)
                   rename("xyz/rbcs.xyz", "xyz/rbcs-equilibration.xyz");

                if(access("xyz/ctcs-equilibration.xyz", F_OK ) == -1)
                    rename("xyz/ctcs.xyz", "xyz/ctcs-equilibration.xyz");
            }

            MPI_CHECK(MPI_Barrier(activecomm));

            wallcreated = true;
	    }

        // note: AMPI's MPIO hangs when appending, so we never append here...
	    xyz_dump(activecomm, cartcomm, "xyz/particles.xyz", "all-particles", p, n, false);
	}

    CUDA_CHECK(cudaEventDestroy(evdownloaded));
}
#endif // AMPI

#ifndef AMPI
void Simulation::_datadump_async()
{
#ifdef _USE_NVTX_
    nvtxNameOsThread(pthread_self(), "DATADUMP_THREAD");
#endif

    int iddatadump = 0, rank;
    int curr_idtimestep = -1;
    bool wallcreated = false;

    MPI_Comm myactivecomm, mycartcomm;

    MPI_CHECK(MPI_Comm_dup(activecomm, &myactivecomm) );
    MPI_CHECK(MPI_Comm_dup(cartcomm, &mycartcomm) );

    H5PartDump dump_part("allparticles->h5part", activecomm, cartcomm), *dump_part_solvent = NULL;
    H5FieldDump dump_field(globals, cartcomm);

    MPI_CHECK(MPI_Comm_rank(myactivecomm, &rank));

    if (rank == 0)
	mkdir("xyz", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

    MPI_CHECK(MPI_Barrier(myactivecomm));

    while (true)
    {
	pthread_mutex_lock(&mutex_datadump);
	async_thread_initialized = 1;

	while (!datadump_pending)
	    pthread_cond_wait(&request_datadump, &mutex_datadump);

	pthread_mutex_unlock(&mutex_datadump);

	if (curr_idtimestep == datadump_idtimestep)
	    if (simulation_is_done)
		break;

	CUDA_CHECK(cudaEventSynchronize(evdownloaded));

	const int n = particles_datadump.size;
	Particle * p = particles_datadump.data;
	Acceleration * a = accelerations_datadump.data;

	{
	    NVTX_RANGE("diagnostics", NVTX_C1);
	    diagnostics(myactivecomm, mycartcomm, p, n, dt, datadump_idtimestep, a);
	}

	if (globals->xyz_dumps)
	{
	    NVTX_RANGE("xyz dump", NVTX_C2);

	    if (globals->walls && datadump_idtimestep >= globals->wall_creation_stepid && !wallcreated)
	    {
		if (rank == 0)
		{
		    if( access("xyz/particles-equilibration.xyz", F_OK ) == -1 )
			rename ("xyz/particles.xyz", "xyz/particles-equilibration.xyz");

		    if( access( "xyz/rbcs-equilibration.xyz", F_OK ) == -1 )
			rename ("xyz/rbcs.xyz", "xyz/rbcs-equilibration.xyz");

		    if( access( "xyz/ctcs-equilibration.xyz", F_OK ) == -1 )
			rename ("xyz/ctcs.xyz", "xyz/ctcs-equilibration.xyz");
		}

		MPI_CHECK(MPI_Barrier(myactivecomm));

		wallcreated = true;
	    }

	    xyz_dump(myactivecomm, mycartcomm, "xyz/particles->xyz", "all-particles", p, n, datadump_idtimestep > 0);
	}

	if (globals->hdf5part_dumps)
	{
	    NVTX_RANGE("h5part dump", NVTX_C3);

	    if (!dump_part_solvent && globals->walls && datadump_idtimestep >= globals->wall_creation_stepid)
	    {
		dump_part.close();

		dump_part_solvent = new H5PartDump("solvent-particles->h5part", activecomm, cartcomm);
	    }

	    if (dump_part_solvent)
		dump_part_solvent->dump(p, n);
	    else
		dump_part.dump(p, n);
	}

	if (globals->hdf5field_dumps)
	{
	    NVTX_RANGE("hdf5 field dump", NVTX_C4);

	    dump_field.dump(activecomm, p, datadump_nsolvent, datadump_idtimestep);
	}

	{
	    NVTX_RANGE("ply dump", NVTX_C5);

	    if (rbcscoll)
		    rbcscoll->dump(myactivecomm, mycartcomm, p + datadump_nsolvent, a + datadump_nsolvent, datadump_nrbcs, iddatadump);

	    if (ctcscoll)
		    ctcscoll->dump(myactivecomm, mycartcomm, p + datadump_nsolvent + datadump_nrbcs,
				    a + datadump_nsolvent + datadump_nrbcs, datadump_nctcs, iddatadump);
	}

	curr_idtimestep = datadump_idtimestep;

	pthread_mutex_lock(&mutex_datadump);

	if (simulation_is_done)
	{
	    pthread_mutex_unlock(&mutex_datadump);
	    break;
	}

	datadump_pending = false;

	pthread_cond_signal(&done_datadump);

	pthread_mutex_unlock(&mutex_datadump);

	++iddatadump;
    }

    if (dump_part_solvent)
	delete dump_part_solvent;

    CUDA_CHECK(cudaEventDestroy(evdownloaded));
}
#endif // AMPI

void Simulation::_update_and_bounce()
{
    double tstart = MPI_Wtime();
    particles->update_stage2_and_1(driving_acceleration, mainstream);

    CUDA_CHECK(cudaPeekAtLastError());

    if (rbcscoll)
	rbcscoll->update_stage2_and_1(driving_acceleration, mainstream);

    CUDA_CHECK(cudaPeekAtLastError());

    if (ctcscoll)
	ctcscoll->update_stage2_and_1(driving_acceleration, mainstream);
    AMPI_YIELD(activecomm);

    timings["update"] += MPI_Wtime() - tstart;

    if (wall.is_active())
    {
	tstart = MPI_Wtime();
	wall.bounce(particles->xyzuvw.data, particles->size, mainstream);

	if (rbcscoll)
	    wall.bounce(rbcscoll->data(), rbcscoll->pcount(), mainstream);

	if (ctcscoll)
	    wall.bounce(ctcscoll->data(), ctcscoll->pcount(), mainstream);
    AMPI_YIELD(activecomm);

	timings["bounce-walls"] += MPI_Wtime() - tstart;
    }

    CUDA_CHECK(cudaPeekAtLastError());
}

Simulation::Simulation(Globals* globals, MPI_Comm cartcomm, MPI_Comm activecomm, bool (*check_termination)()) :
    GlobalsInjector(globals), cartcomm(cartcomm), activecomm(activecomm),
    /*particles(_ic()),*/ cells(globals, XSIZE_SUBDOMAIN, YSIZE_SUBDOMAIN, ZSIZE_SUBDOMAIN),
    rbcscoll(NULL), ctcscoll(NULL), wall(globals, cartcomm),
    redistribute(cartcomm),  redistribute_rbcs(globals, cartcomm),  redistribute_ctcs(globals, cartcomm),
    /*dpd(globals, cartcomm), fsi(cartcomm), contact(cartcomm), solutex(cartcomm),*/
    dpd(NULL), fsi(NULL), contact(NULL), solutex(NULL),
    check_termination(check_termination),
    driving_acceleration(0), host_idle_time(0), nsteps((int)(globals->tend / dt)),
#ifndef AMPI
    datadump_pending(false),
#endif
    simulation_is_done(false),
    report_t0_a(MPI_Wtime()), report_t0_b(MPI_Wtime())
{
    particles_pingpong[0].globals = globals;
    particles_pingpong[1].globals = globals;

    MPI_CHECK( MPI_Comm_size(activecomm, &nranks) );
    MPI_CHECK( MPI_Comm_rank(activecomm, &rank) );

    _post_migrate();
    /*solutex->attach_halocomputation(fsi);

    if (globals->contactforces)
	solutex->attach_halocomputation(contact);*/
    //globals->localcomm.initialize(activecomm);

    int dims[3], periods[3], coords[3];
    MPI_CHECK( MPI_Cart_get(cartcomm, 3, dims, periods, coords) );

    {
	particles = &particles_pingpong[0];
	newparticles = &particles_pingpong[1];

	vector<Particle> ic = _ic();

	for(int c = 0; c < 2; ++c)
	{
	    particles_pingpong[c].resize(ic.size());

	    particles_pingpong[c].origin = make_float3((0.5 + coords[0]) * XSIZE_SUBDOMAIN,
						       (0.5 + coords[1]) * YSIZE_SUBDOMAIN,
						       (0.5 + coords[2]) * ZSIZE_SUBDOMAIN);

	    particles_pingpong[c].globalextent = make_float3(dims[0] * XSIZE_SUBDOMAIN,
							     dims[1] * YSIZE_SUBDOMAIN,
							     dims[2] * ZSIZE_SUBDOMAIN);
	}

	CUDA_CHECK(cudaMemcpy(particles->xyzuvw.data, &ic.front(), sizeof(Particle) * ic.size(), cudaMemcpyHostToDevice));

	cells.build(particles->xyzuvw.data, particles->size, 0, NULL, NULL);

	_update_helper_arrays();
    }

    if (globals->rbcs)
    {
	rbcscoll = new CollectionRBC(globals, cartcomm);
	rbcscoll->setup("rbcs-ic.txt");
    }

    if (globals->ctcs)
    {
	ctcscoll = new CollectionCTC(globals, cartcomm);
	ctcscoll->setup("ctcs-ic.txt");
    }

#ifndef _NO_DUMPS_
    //setting up the asynchronous data dumps
    {
#ifndef AMPI
	CUDA_CHECK(cudaEventCreate(&evdownloaded, cudaEventDisableTiming | cudaEventBlockingSync));
#endif

	particles_datadump.resize(particles->size * 1.5);
	accelerations_datadump.resize(particles->size * 1.5);

#ifndef AMPI
	int rc = pthread_mutex_init(&mutex_datadump, NULL);
	rc |= pthread_cond_init(&done_datadump, NULL);
	rc |= pthread_cond_init(&request_datadump, NULL);
	async_thread_initialized = 0;
	rc |= pthread_create(&thread_datadump, NULL, datadump_trampoline, this);

	while (1)
	{
	    pthread_mutex_lock(&mutex_datadump);
	    int done = async_thread_initialized;
	    pthread_mutex_unlock(&mutex_datadump);

	    if (done)
		break;
	}

	if (rc)
	{
	    printf("ERROR; return code from pthread_create() is %d\n", rc);
	    exit(-1);
	}
#endif // AMPI
    }
#endif
}

void Simulation::_lockstep()
{
    double tstart = MPI_Wtime();

    SolventWrap wsolvent(particles->xyzuvw.data, particles->size, particles->axayaz.data, cells.start, cells.count);

    std::vector<ParticlesWrap> wsolutes;

    if (rbcscoll)
	wsolutes.push_back(ParticlesWrap(rbcscoll->data(), rbcscoll->pcount(), rbcscoll->acc()));

    if (ctcscoll)
	wsolutes.push_back(ParticlesWrap(ctcscoll->data(), ctcscoll->pcount(), ctcscoll->acc()));

    fsi->bind_solvent(wsolvent);

    solutex->bind_solutes(wsolutes);

    particles->clear_acc(mainstream);

    if (rbcscoll)
	rbcscoll->clear_acc(mainstream);

    if (ctcscoll)
	ctcscoll->clear_acc(mainstream);

    solutex->pack_p(mainstream);

    dpd->pack(particles->xyzuvw.data, particles->size, cells.start, cells.count, mainstream);

    dpd->local_interactions(particles->xyzuvw.data, xyzouvwo.data, xyzo_half.data, particles->size, particles->axayaz.data,
			   cells.start, cells.count, mainstream);

    if (globals->contactforces)
	contact->build_cells(wsolutes, mainstream);

    solutex->post_p(mainstream, downloadstream);

    dpd->post(particles->xyzuvw.data, particles->size, mainstream, downloadstream);

    CUDA_CHECK(cudaPeekAtLastError());

    if (wall.is_active())
	wall.interactions(particles->xyzuvw.data, particles->size, particles->axayaz.data,
			   cells.start, cells.count, mainstream);

    CUDA_CHECK(cudaPeekAtLastError());

    dpd->recv(mainstream, uploadstream);

    solutex->recv_p(uploadstream);

    solutex->halo(uploadstream, mainstream);

    dpd->remote_interactions(particles->xyzuvw.data, particles->size, particles->axayaz.data, mainstream, uploadstream);

    fsi->bulk(wsolutes, mainstream);

    if (globals->contactforces)
	contact->bulk(wsolutes, mainstream);

    CUDA_CHECK(cudaPeekAtLastError());

    if (rbcscoll)
	CudaRBC::forces_nohost(mainstream, rbcscoll->count(), (float *)rbcscoll->data(), (float *)rbcscoll->acc());

    if (ctcscoll)
	CudaCTC::forces_nohost(mainstream, ctcscoll->count(), (float *)ctcscoll->data(), (float *)ctcscoll->acc());

    CUDA_CHECK(cudaPeekAtLastError());

    solutex->post_a();

    particles->update_stage2_and_1(driving_acceleration, mainstream);

    if (wall.is_active())
	wall.bounce(particles->xyzuvw.data, particles->size, mainstream);

    CUDA_CHECK(cudaPeekAtLastError());

    redistribute.pack(particles->xyzuvw.data, particles->size, mainstream);

    redistribute.send();

    redistribute.bulk(particles->size, cells.start, cells.count, mainstream);

    CUDA_CHECK(cudaPeekAtLastError());

    if (rbcscoll && wall.is_active())
	wall.interactions(rbcscoll->data(), rbcscoll->pcount(), rbcscoll->acc(), NULL, NULL, mainstream);

    if (ctcscoll && wall.is_active())
	wall.interactions(ctcscoll->data(), ctcscoll->pcount(), ctcscoll->acc(), NULL, NULL, mainstream);

    CUDA_CHECK(cudaPeekAtLastError());

    solutex->recv_a(mainstream);

    if (rbcscoll)
	rbcscoll->update_stage2_and_1(driving_acceleration, mainstream);

    if (ctcscoll)
	ctcscoll->update_stage2_and_1(driving_acceleration, mainstream);

    if (wall.is_active() && rbcscoll)
	wall.bounce(rbcscoll->data(), rbcscoll->pcount(), mainstream);

    if (wall.is_active() && ctcscoll)
	wall.bounce(ctcscoll->data(), ctcscoll->pcount(), mainstream);

    const int newnp = redistribute.recv_count(mainstream, host_idle_time);

    CUDA_CHECK(cudaPeekAtLastError());

    if (rbcscoll)
	redistribute_rbcs.extent(rbcscoll->data(), rbcscoll->count(), mainstream);

    if (ctcscoll)
	redistribute_ctcs.extent(ctcscoll->data(), ctcscoll->count(), mainstream);

    if (rbcscoll)
	redistribute_rbcs.pack_sendcount(rbcscoll->data(), rbcscoll->count(), mainstream);

    if (ctcscoll)
	redistribute_ctcs.pack_sendcount(ctcscoll->data(), ctcscoll->count(), mainstream);

    newparticles->resize(newnp);
    xyzouvwo.resize(newnp * 2);
    xyzo_half.resize(newnp);

    redistribute.recv_unpack(newparticles->xyzuvw.data, xyzouvwo.data, xyzo_half.data, newnp, cells.start, cells.count, mainstream, host_idle_time);

    CUDA_CHECK(cudaPeekAtLastError());

    swap(particles, newparticles);

    int nrbcs;
    if (rbcscoll)
	nrbcs = redistribute_rbcs.post();

    int nctcs;
    if (ctcscoll)
	nctcs = redistribute_ctcs.post();

    if (rbcscoll)
	rbcscoll->resize(nrbcs);

    if (ctcscoll)
	ctcscoll->resize(nctcs);

    CUDA_CHECK(cudaPeekAtLastError());

    if (rbcscoll)
	redistribute_rbcs.unpack(rbcscoll->data(), rbcscoll->count(), mainstream);

    if (ctcscoll)
	redistribute_ctcs.unpack(ctcscoll->data(), ctcscoll->count(), mainstream);

    CUDA_CHECK(cudaPeekAtLastError());

    timings["lockstep"] += MPI_Wtime() - tstart;
}

void Simulation::_pre_migrate()
{
    CUDA_CHECK(cudaDeviceSynchronize());

    if (dpd)
        delete dpd;
    dpd = NULL;
    if (solutex)
        delete solutex;
    solutex = NULL;
    if (fsi)
        delete fsi;
    fsi = NULL;
    if (contact)
        delete contact;
    contact = NULL;

    if (wall.is_active())
        wall.destroy_sdf_texture();

    CUDA_CHECK(cudaStreamDestroy(mainstream));
    CUDA_CHECK(cudaStreamDestroy(uploadstream));
    CUDA_CHECK(cudaStreamDestroy(downloadstream));
}

void Simulation::_post_migrate()
{
    dpd = new ComputeDPD(globals, cartcomm);
    solutex = new SoluteExchange(cartcomm);
    fsi = new ComputeFSI(cartcomm);
    contact = new ComputeContact(cartcomm);

    solutex->attach_halocomputation(*fsi);
    if (globals->contactforces)
        solutex->attach_halocomputation(*contact);

    CUDA_CHECK(cudaStreamCreate(&mainstream));
    CUDA_CHECK(cudaStreamCreate(&uploadstream));
    CUDA_CHECK(cudaStreamCreate(&downloadstream));

    redistribute.update_device_pointers();
    if (wall.is_active())
        wall.create_sdf_texture();
}

void Simulation::_migrate()
{
#ifdef AMPI
    const double tstart = MPI_Wtime();
    const int pe_before = MPI_My_pe();
    int vp;
    MPI_Comm_rank(MPI_COMM_WORLD, &vp);
    _pre_migrate();
    MPI_Migrate();
    _post_migrate();
    const int pe_after = MPI_My_pe();
    if (pe_before != pe_after)
        printf("\x1b[95mmigrated VP %d from PE %d to PE %d\x1b[0m\n", vp, pe_before, pe_after);
    timings["migration"] += MPI_Wtime() - tstart;
#endif
}


void Simulation::run()
{
    if (rank == 0 && !globals->walls)
	printf("the simulation begins now and it consists of %.3e steps\n", (double)nsteps);

    double time_simulation_start = MPI_Wtime();

    _redistribute();
    _forces();

    if (!globals->walls && globals->pushtheflow)
	driving_acceleration = hydrostatic_a;

    particles->update_stage1(driving_acceleration, mainstream);

    if (rbcscoll)
	rbcscoll->update_stage1(driving_acceleration, mainstream);

    if (ctcscoll)
	ctcscoll->update_stage1(driving_acceleration, mainstream);
    AMPI_YIELD(activecomm);

    int it;


    for(it = 0; it < nsteps; ++it)
    {
	const bool verbose = it > 0 && rank == 0;

#ifdef _USE_NVTX_
	if (it == globals->nvtxstart)
	{
	    NvtxTracer::currently_profiling = true;
	    CUDA_CHECK(cudaProfilerStart());
	}
	else if (it == globals->nvtxstop)
	{
	    CUDA_CHECK(cudaProfilerStop());
	    NvtxTracer::currently_profiling = false;
	    CUDA_CHECK(cudaDeviceSynchronize());

	    if (rank == 0)
		printf("profiling session ended. terminating the simulation now...\n");

	    break;
	}
#endif

	if (it % globals->steps_per_report == 0)
	{
	    CUDA_CHECK(cudaStreamSynchronize(mainstream));

	    if (simulation_is_done = check_termination())
		break;

	    _report(verbose, it);
	}

	_redistribute();

#if 1
    lockstep_check:

	const bool lockstep_OK =
	    !(globals->walls && it >= globals->wall_creation_stepid && !wall.is_active()) &&
	    !(it % globals->steps_per_dump == 0) &&
	    !(it + 1 == globals->nvtxstart) &&
	    !(it + 1 == globals->nvtxstop) &&
	    !((it + 1) % globals->steps_per_report == 0) &&
	    !(it + 1 == nsteps);
    const bool next_lockstep_OK = 
	    !(globals->walls && (it + 1) >= globals->wall_creation_stepid && !wall.is_active()) &&
	    !((it + 1) % globals->steps_per_dump == 0) &&
	    !((it + 1) + 1 == globals->nvtxstart) &&
	    !((it + 1) + 1 == globals->nvtxstop) &&
	    !(((it + 1) + 1) % globals->steps_per_report == 0) &&
	    !((it + 1) + 1 == nsteps);
#ifdef AMPI
    const int synchronous_steps = globals->steps_per_report / 10;
    const bool synchronous = it % globals->steps_per_report < synchronous_steps;

    if (synchronous && mainstream) {
        CUDA_CHECK(cudaStreamDestroy(mainstream));
        mainstream = 0;
        MPI_Start_measure();
    } else if (!synchronous && !mainstream) {
        CUDA_CHECK(cudaStreamCreate(&mainstream));
        MPI_Stop_measure();
    }
#endif

	if (lockstep_OK)
	{
        if (!next_lockstep_OK) {
            redistribute.set_lastcall();
            dpd->set_lastpost();
        } 
	    _lockstep();

	    ++it;

	    goto lockstep_check;
    }
    if (redistribute.migratable() && it > globals->steps_per_report)
        _migrate();
#endif

	if (globals->walls && it >= globals->wall_creation_stepid && !wall.is_active())
	{
	    CUDA_CHECK(cudaDeviceSynchronize());

	    bool termination_request = false;

	    _create_walls(verbose, termination_request);

	    _redistribute();

	    if (termination_request)
		break;

	    time_simulation_start = MPI_Wtime();

	    if (globals->pushtheflow)
		driving_acceleration = hydrostatic_a;

	    if (rank == 0)
		printf("the simulation begins now and it consists of %.3e steps\n", (double)(nsteps - it));
	}

	_forces();

#ifndef _NO_DUMPS_
	if (it % globals->steps_per_dump == 0)
	    _datadump(it);
#endif
	_update_and_bounce();

    }

    const double time_simulation_stop = MPI_Wtime();
    const double telapsed = time_simulation_stop - time_simulation_start;

    simulation_is_done = true;

    if (rank == 0)
	if (it == nsteps)
	    printf("simulation is done after %.2lf s (%dm%ds). Ciao.\n",
		   telapsed, (int)(telapsed / 60), (int)(telapsed) % 60);
	else
	    if (it != globals->wall_creation_stepid)
		printf("external termination request (signal) after %.3e s. Bye.\n", telapsed);

    fflush(stdout);
}

Simulation::~Simulation()
{
#ifndef AMPI
#ifndef _NO_DUMPS_
    pthread_mutex_lock(&mutex_datadump);

    datadump_pending = true;
    pthread_cond_signal(&request_datadump);

    pthread_mutex_unlock(&mutex_datadump);

    pthread_join(thread_datadump, NULL);
#endif
#endif // AMPI

    CUDA_CHECK(cudaStreamDestroy(mainstream));
    CUDA_CHECK(cudaStreamDestroy(uploadstream));
    CUDA_CHECK(cudaStreamDestroy(downloadstream));

    if (dpd)
    delete dpd;

    if (solutex)
    delete solutex;

    if (fsi)
    delete fsi;

    if (contact)
    delete contact;

    if (rbcscoll)
	delete rbcscoll;

    if (ctcscoll)
	delete ctcscoll;
}
