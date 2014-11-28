#include <cstring>
#include <algorithm>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>

#include "halo-exchanger.h"

using namespace std;

HaloExchanger::HaloExchanger(MPI_Comm _cartcomm, int L):  L(L)
{
    assert(L % 2 == 0);
    assert(L >= 2);

    MPI_CHECK( MPI_Comm_dup(_cartcomm, &cartcomm));

    MPI_CHECK( MPI_Comm_rank(cartcomm, &myrank));
    MPI_CHECK( MPI_Comm_size(cartcomm, &nranks));

    MPI_CHECK( MPI_Cart_get(cartcomm, 3, dims, periods, coords) );

    for(int i = 0; i < 26; ++i)
    {
	int d[3] = { (i + 2) % 3 - 1, (i / 3 + 2) % 3 - 1, (i / 9 + 2) % 3 - 1 };

	recv_tags[i] = (2 - d[0]) % 3 + 3 * ((2 - d[1]) % 3 + 3 * ((2 - d[2]) % 3));

	int coordsneighbor[3];
	for(int c = 0; c < 3; ++c)
	    coordsneighbor[c] = coords[c] + d[c];

	MPI_CHECK( MPI_Cart_rank(cartcomm, coordsneighbor, dstranks + i) );

	const int nhalocells = pow(L, 3 - fabs(d[0]) - fabs(d[1]) - fabs(d[2]));
	const int estimate = 2 * 3 * nhalocells;

	halosize[i].x = d[0] != 0 ? 1 : L;
	halosize[i].y = d[1] != 0 ? 1 : L;
	halosize[i].z = d[2] != 0 ? 1 : L;
	assert(nhalocells == halosize[i].x * halosize[i].y * halosize[i].z);
	
	recvhalos[i].buf.resize(estimate);
	recvhalos[i].cellstarts.resize(nhalocells + 1);
	
	sendhalos[i].buf.resize(estimate);
	sendhalos[i].scattered_entries.resize(estimate);
	sendhalos[i].cellstarts.resize(nhalocells + 1);
	sendhalos[i].tmpcount.resize(nhalocells + 1);
	sendhalos[i].tmpstart.resize(nhalocells + 1);
    }

    CUDA_CHECK(cudaHostAlloc((void **)&required_send_bag_size, sizeof(int) * 26, cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(&required_send_bag_size_host, required_send_bag_size, 0));

    for(int i = 0; i < 7; ++i)
	CUDA_CHECK(cudaStreamCreate(streams + i));

    for(int i = 0, ctr = 1; i < 26; ++i)
    {
	int d[3] = { (i + 2) % 3 - 1, (i / 3 + 2) % 3 - 1, (i / 9 + 2) % 3 - 1 };

	const bool isface = abs(d[0]) + abs(d[1]) + abs(d[2]) == 1;

	code2stream[i] = 0;

	if (isface)
	{
	    code2stream[i] = ctr;
	    ctr++;
	}
    }
}

namespace PackingHalo
{
    __device__ int blockcount, global_histo[27], requiredsize;

    __global__ void setup()
    {
	blockcount = 0;
	requiredsize = 0;

	for(int i = 0; i < 27; ++i)
	    global_histo[i] = 0;
    }

    template< int work >
    __global__ void count(int * const packs_start, const Particle * const p, const int np, const int L, 
			  int * bag_size_required)
    {
	assert(blockDim.x * gridDim.x * work >= np);
	assert(blockDim.x >= 26);

	__shared__ int histo[26];

	const int tid = threadIdx.x; 

	if (tid < 26)
	    histo[tid] = 0;

	__syncthreads();

	for(int t = 0; t < work; ++t)
	{
	    const int pid = tid + blockDim.x * (blockIdx.x + gridDim.x * t);

	    if (pid < np)
		for(int i = 0; i < 26; ++i)
		{
		    int d[3] = { (i + 2) % 3 - 1, (i / 3 + 2) % 3 - 1, (i / 9 + 2) % 3 - 1 };

		    bool halo = true;			

		    for(int c = 0; c < 3; ++c)
		    {
			const float halo_start = max(d[c] * L - L/2 - 1, -L/2);
			const float halo_end = min(d[c] * L + L/2 + 1, L/2);

			const float x = p[pid].x[c];

			halo &= (x >= halo_start && x < halo_end);
		    }

		    if (halo)
			atomicAdd(histo + i, 1);
		}
	}

	__syncthreads();

	if (tid < 26 && histo[tid] > 0)
	    atomicAdd(global_histo + tid, histo[tid]);

	if (tid == 0)
	{
	    const int timestamp = atomicAdd(&blockcount, 1);

	    if (timestamp == gridDim.x - 1)
	    {
		blockcount = 0;

		int s = 0, curr;

		for(int i = 0; i < 26; ++i)
		{
		    curr = global_histo[i];
		    global_histo[i] = packs_start[i] = s;
		    s += curr;
		}

		global_histo[26] = packs_start[26] = s;
		requiredsize = s;		
		*bag_size_required = s;
	    }
	}
    }

    __global__ void count(const int * const cellsstart, const int * const cellscount,
			  const int3 halo_offset, const int3 halo_size, const int L,
			  int * const output_start, int * const output_count)
    {
	assert(halo_size.x * halo_size.y * halo_size.z <= blockDim.x * gridDim.x);

	const int gid = threadIdx.x + blockDim.x * blockIdx.x;

	const int3 tmp = make_int3(gid % halo_size.x, (gid / halo_size.x) % halo_size.y, (gid / (halo_size.x * halo_size.y)));
	 
	const int3 dst = make_int3(halo_offset.x + tmp.x,
				   halo_offset.y + tmp.y,
				   halo_offset.z + tmp.z);

	const int nsize = halo_size.x * halo_size.y * halo_size.z;
	 
	if (gid < nsize)
	{
	    assert(dst.x >= 0 && dst.x < L);
	    assert(dst.y >= 0 && dst.y < L);
	    assert(dst.z >= 0 && dst.z < L);
	    
	    const int srcentry = dst.x + L * (dst.y + L * dst.z);

	    assert(srcentry < L * L * L);

	    output_start[gid] = cellsstart[srcentry];
	    output_count[gid] = cellscount[srcentry];
	}
	else
	    if (gid == nsize)
		output_start[gid] = output_count[gid] = 0;
    }

    __global__ void fill(const Particle * const particles, const int np,
			 const int * const start_src, const int * const count_src, const int L,
			 const int * const start_dst, 
			 Particle * const bag, const int bagsize, int * const scattered_entries, int * const required_bag_size, const int code)
    {
	assert(blockDim.x == warpSize);

	const int cellid = blockIdx.x;
	const int tid = threadIdx.x;

	const int base_src = start_src[cellid];
	const int base_dst = start_dst[cellid];

	const int nsrc = min(count_src[cellid], bagsize - base_dst);
	
	for(int i = tid; i < nsrc; i += warpSize)
	{
	    const int pid = base_src + i;

	    assert(pid < np && pid >= 0);
	    
	    bag[base_dst + i] = particles[pid];
	    scattered_entries[base_dst + i] = pid;

#ifndef NDEBUG
	    {
		int d[3] = { (code + 2) % 3 - 1, (code / 3 + 2) % 3 - 1, (code / 9 + 2) % 3 - 1 };
		
		for(int c = 0; c < 3; ++c)
		{
		    const float halo_start = max(d[c] * L - L/2 - 1, -L/2);
		    const float halo_end = min(d[c] * L + L/2 + 1, L/2);

		    if (!(particles[pid].x[c] >= halo_start && particles[pid].x[c] < halo_end))
		    {
			printf("oooops particle %d: %e %e %e component %d not within %f , %f\n", pid, particles[pid].x[0], particles[pid].x[1], particles[pid].x[2],
			       c, halo_start, halo_end);
		
		    }
		    const float eps = 1e-5;
		    assert(particles[pid].x[c] >= halo_start - eps && particles[pid].x[c] < halo_end + eps);
		}
	    }
#endif
	}

	if (cellid == gridDim.x - 1)
	    *required_bag_size = base_dst;
    }
     
    __global__ void pack(const Particle * const particles, const int np, const int L, Particle * const bag, 
			 const int bagsize, int * const scattered_entries)
    {
	if (bagsize < requiredsize)
	    return;

	assert(blockDim.x * gridDim.x >= np);
	assert(blockDim.x >= 26);

	__shared__ int histo[26];
	__shared__ int base[26];

	const int tid = threadIdx.x; 

	if (tid < 26)
	    histo[tid] = 0;

	__syncthreads();

	int offset[26];
	for(int i = 0; i < 26; ++i)
	    offset[i] = -1;

	Particle p;

	const int pid = tid + blockDim.x * blockIdx.x;

	if (pid < np)
	{
	    p = particles[pid];

	    for(int c = 0; c < 3; ++c)
		assert(p.x[c] >= -L / 2 && p.x[c] < L / 2);

	    for(int i = 0; i < 26; ++i)
	    {
		int d[3] = { (i + 2) % 3 - 1, (i / 3 + 2) % 3 - 1, (i / 9 + 2) % 3 - 1 };

		bool halo = true;			

		for(int c = 0; c < 3; ++c)
		{
		    const float halo_start = max(d[c] * L - L/2 - 1, -L/2);
		    const float halo_end = min(d[c] * L + L/2 + 1, L/2);

		    const float x = p.x[c];

		    halo &= (x >= halo_start && x < halo_end);
		}

		if (halo)
		    offset[i] = atomicAdd(histo + i, 1);
	    }
	}
	__syncthreads();

	if (tid < 26 && histo[tid] > 0)
	    base[tid] = atomicAdd(global_histo + tid, histo[tid]);

	__syncthreads();

	for(int i = 0; i < 26; ++i)
	    if (offset[i] != -1)
	    {
		const int entry = base[i] + offset[i];
		assert(entry >= 0 && entry < global_histo[26]); 

		bag[ entry ] = p; 
		scattered_entries[ entry ] = pid;
	    }
    }

    __global__ void shift_recv_particles(Particle * p, int n, int L, int code)
    {
	assert(blockDim.x * gridDim.x >= n);

	const int pid = threadIdx.x + blockDim.x * blockIdx.x;

	if (pid >= n)
	    return;

	for(int c = 0; c < 3; ++c)
	    assert(p[pid].x[c] >= -L / 2 && p[pid].x[c] < L / 2);

	const int d[3] = { (code + 2) % 3 - 1, (code / 3 + 2) % 3 - 1, (code / 9 + 2) % 3 - 1 };

	for(int c = 0; c < 3; ++c)
	    p[pid].x[c] += d[c] * L;

#ifndef NDEBUG

	assert(p[pid].x[0] <= -L / 2 || p[pid].x[0] >= L / 2 ||
	       p[pid].x[1] <= -L / 2 || p[pid].x[1] >= L / 2 || 
	       p[pid].x[2] <= -L / 2 || p[pid].x[2] >= L / 2);

	for(int c = 0; c < 3; ++c)
	{
	    const float halo_start = max(d[c] * L - L/2, -L/2 - 1);
	    const float halo_end = min(d[c] * L + L/2, L/2 + 1);
	    const float eps = 1e-5;
	    assert(p[pid].x[c] >= halo_start - eps && p[pid].x[c] <= halo_end + eps);
	}

#endif
    }

    __global__ void check_send_particles(Particle * p, int n, int L, int code)
    {
	assert(blockDim.x * gridDim.x >= n);

	const int pid = threadIdx.x + blockDim.x * blockIdx.x;

	if (pid >= n)
	    return;

	assert(p[pid].x[0] >= -L / 2 || p[pid].x[0] < L / 2 ||
	       p[pid].x[1] >= -L / 2 || p[pid].x[1] < L / 2 || 
	       p[pid].x[2] >= -L / 2 || p[pid].x[2] < L / 2);

	const int d[3] = { (code + 2) % 3 - 1, (code / 3 + 2) % 3 - 1, (code / 9 + 2) % 3 - 1 };

	for(int c = 0; c < 3; ++c)
	{
	    const float halo_start = max(d[c] * L - L/2 - 1, -L/2);
	    const float halo_end = min(d[c] * L + L/2 + 1, L/2);

	    if (!(p[pid].x[c] >= halo_start && p[pid].x[c] < halo_end))
	    {
		printf("oooops particle %d: %e %e %e component %d not within %f , %f\n", pid, p[pid].x[0], p[pid].x[1], p[pid].x[2],
		       c, halo_start, halo_end);
		
	    }
	    const float eps = 1e-5;
	    assert(p[pid].x[c] >= halo_start - eps && p[pid].x[c] < halo_end + eps);
	}
    }
}

void HaloExchanger::pack_and_post(const Particle * const p, const int n, const int * const cellsstart, const int * const cellscount)
{
    nlocal = n;

    for(int i = 0; i < 26; ++i)
    {
	int d[3] = { (i + 2) % 3 - 1, (i / 3 + 2) % 3 - 1, (i / 9 + 2) % 3 - 1 };
	 
	int halo_start[3], halo_size[3];
	for(int c = 0; c < 3; ++c)
	{
	    halo_start[c] = max(d[c] * L - L/2 - 1, -L/2);
	    halo_size[c] = min(d[c] * L + L/2 + 1, L/2) - halo_start[c];
	}

	const int nentries = sendhalos[i].cellstarts.size;
	
	PackingHalo::count<<< (nentries + 127) / 128, 128, 0, streams[code2stream[i]] >>>
	    (cellsstart, cellscount,  make_int3(halo_start[0] + L/2 , halo_start[1] + L/2, halo_start[2] + L/2),
	     make_int3(halo_size[0], halo_size[1], halo_size[2]), L, sendhalos[i].tmpstart.data, sendhalos[i].tmpcount.data);
    }
    
    CUDA_CHECK(cudaPeekAtLastError());

    /* replace this exclusive scan with a lightweight home-made version */
    for(int i = 0; i < 26; ++i)
	thrust::exclusive_scan(thrust::device_ptr<int>(sendhalos[i].tmpcount.data),
			       thrust::device_ptr<int>(sendhalos[i].tmpcount.data + sendhalos[i].tmpcount.size),
			       thrust::device_ptr<int>(sendhalos[i].cellstarts.data));

    CUDA_CHECK(cudaPeekAtLastError());
  	
    for(int pass = 0; pass < 2; ++pass)
    {
	bool needsync = pass == 0;

	for(int i = 0; i < 26; ++i)
	{
	    bool fail = false;
	    int nrequired;

	    if (pass == 1)
	    {
		nrequired = required_send_bag_size_host[i];
		fail = sendhalos[i].buf.capacity < nrequired;
	    }

	    if (pass == 0 || fail)
	    {
		if (fail)
		{
		    printf("------------------- rank %d - code %d : oops now: %d required: %d\n", myrank, i, sendhalos[i].buf.size, nrequired);
		    sendhalos[i].buf.resize(nrequired);
		    sendhalos[i].scattered_entries.resize(nrequired);
		    needsync = true;
		}
		
		const int nentries = sendhalos[i].cellstarts.size;

		PackingHalo::fill<<<nentries, 32, 0, streams[code2stream[i]] >>>
		    (p, n, sendhalos[i].tmpstart.data, sendhalos[i].tmpcount.data, L, sendhalos[i].cellstarts.data,
		     sendhalos[i].buf.data, sendhalos[i].buf.capacity, sendhalos[i].scattered_entries.data, required_send_bag_size + i, i);
	    }

	    if (pass == 1)
	    {
		sendhalos[i].buf.size = nrequired;
		sendhalos[i].scattered_entries.size = nrequired;
	    }
	}

	CUDA_CHECK(cudaPeekAtLastError());

	if (needsync)
	    for(int i = 0; i < 7; ++i)
	    CUDA_CHECK(cudaStreamSynchronize(streams[i]));
    }
	
  #ifndef NDEBUG
    for(int i = 0; i < 26; ++i)
    {
    	const int nd = sendhalos[i].buf.size;
	
	if (nd > 0)
	    PackingHalo::check_send_particles<<<(nd + 127)/ 128, 128>>>(sendhalos[i].buf.data, nd, L, i);
    }

    CUDA_CHECK(cudaStreamSynchronize(0));
    CUDA_CHECK(cudaPeekAtLastError());
#endif

    for(int i = 0; i < 26; ++i)
	MPI_CHECK( MPI_Irecv(recvhalos[i].cellstarts.data, recvhalos[i].cellstarts.size, MPI_INTEGER,
			     dstranks[i], recv_tags[i] + 350, cartcomm,  recvcellsreq + i) );
    
    for(int i = 0; i < 26; ++i)
	MPI_CHECK( MPI_Isend(sendhalos[i].cellstarts.data, sendhalos[i].cellstarts.size, MPI_INTEGER,
			     dstranks[i], i + 350, cartcomm,  sendcellsreq + i) );
    
    //retrieve recv_counts
    {
	MPI_Request sendcountreq[26];

	for(int i = 0; i < 26; ++i)
	    MPI_CHECK( MPI_Isend(&sendhalos[i].buf.size, 1, MPI_INTEGER, dstranks[i],  i+ 150, cartcomm, sendcountreq + i) );
	
	MPI_Status status;

	int sum = 0;
	for(int i = 0; i < 26; ++i)
	{
	int count;
	    MPI_CHECK( MPI_Recv(&count, 1, MPI_INTEGER, dstranks[i],  recv_tags[i] + 150, cartcomm, &status) );

	    recvhalos[i].buf.resize(count);

	    sum += count;
	}

	MPI_Status culo[26];
	MPI_CHECK( MPI_Waitall(26, sendcountreq, culo) );
    }

    nrecvreq = 0;

    for(int i = 0; i < 26; ++i)
    {
	const int count = recvhalos[i].buf.size;

	if (count == 0)
	    continue;

	MPI_CHECK( MPI_Irecv(recvhalos[i].buf.data, count, Particle::datatype(), dstranks[i], 
			     recv_tags[i], cartcomm, recvreq + nrecvreq) );	

	++nrecvreq;
    }

    nsendreq = 0;

    for(int i = 0; i < 26; ++i)
    {
	const int count = sendhalos[i].buf.size;

	if (count == 0) 
	    continue;

	MPI_CHECK( MPI_Isend(sendhalos[i].buf.data, count, Particle::datatype(), dstranks[i], 
			     i, cartcomm, sendreq + nsendreq) );

	++nsendreq;
    }
}

void HaloExchanger::wait_for_messages()
{
    {
	MPI_Status statuses[26];

	MPI_CHECK( MPI_Waitall(nrecvreq, recvreq, statuses) );    
	MPI_CHECK( MPI_Waitall(nsendreq, sendreq, statuses) );
	MPI_CHECK( MPI_Waitall(26, recvcellsreq, statuses) );
	MPI_CHECK( MPI_Waitall(26, sendcellsreq, statuses) );
    }

    for(int i = 0; i < 26; ++i)
    {
	const int ns = recvhalos[i].buf.size;

	if (ns > 0)
	    PackingHalo::shift_recv_particles<<<(ns + 127) / 128, 128, 0, streams[code2stream[i]]>>>(recvhalos[i].buf.data, ns, L, i);	
    }
}

int HaloExchanger::nof_sent_particles()
{
    int s = 0;
    for(int i = 0; i < 26; ++i)
	s += sendhalos[i].buf.size;

    return s;
}

SimpleDeviceBuffer<Particle> HaloExchanger::exchange(const Particle * const plocal, int nlocal,
	    const int * const cellsstart, const int * const cellscount)
{
    pack_and_post(plocal, nlocal, cellsstart, cellscount);
    wait_for_messages();

    int s = 0;
    for(int i = 0; i < 26; ++i)
	s += recvhalos[i].buf.size;
    
    SimpleDeviceBuffer<Particle> retval(s);

    s = 0;
    for(int i = 0; i < 26; ++i)
    {
	CUDA_CHECK(cudaMemcpy(retval.data + s, recvhalos[i].buf.data, recvhalos[i].buf.size * sizeof(Particle), cudaMemcpyDeviceToDevice));
	s += recvhalos[i].buf.size;
    }

    return retval;
}

HaloExchanger::~HaloExchanger()
{
    for(int i = 0; i < 7; ++i)
	CUDA_CHECK(cudaStreamDestroy(streams[i]));
    
    CUDA_CHECK(cudaFreeHost(required_send_bag_size));

    MPI_CHECK(MPI_Comm_free(&cartcomm));
}
