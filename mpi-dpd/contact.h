/*
 *  contact.h
 *  Part of uDeviceX/mpi-dpd/
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

#include "solute-exchange.h"

#include <../dpd-rng.h>

class ComputeContact : public SoluteExchange::Visitor
{
    //cudaEvent_t evuploaded;

    int nsolutes;

    SimpleDeviceBuffer<uchar4> subindices;
    SimpleDeviceBuffer<unsigned char> compressed_cellscount;
    SimpleDeviceBuffer<int> cellsentries, cellsstart, cellscount;

    Logistic::KISS local_trunk;

    // globals moved from contact.cu
    cudaTextureObject_t texCellsStart, texCellEntries;
    int* cnsolutes;
    float2** csolutes;
    float** csolutesacc;
    int* packstarts_padded, *packcount;
    Particle** packstates;
    Acceleration** packresults;
    uint* scan_tmp;
    int ns[32];
    float2* ps[32];
    float* as[32];

public:

    ComputeContact(MPI_Comm comm);
    ~ComputeContact();

    void bind(const int * const cellsstart, const int * const cellsentries, const int ncellentries,
              std::vector<ParticlesWrap> wsolutes, cudaStream_t stream, const int * const cellscount);

    void build_cells(std::vector<ParticlesWrap> wsolutes, cudaStream_t stream);

    void bulk(std::vector<ParticlesWrap> wsolutes, cudaStream_t stream);

    /*override of SoluteExchange::Visitor::halo*/
    void halo(ParticlesWrap solutes[26], cudaStream_t stream);
};
