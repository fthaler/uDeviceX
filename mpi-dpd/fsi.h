/*
 *  rbc-interactions.h
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

class ComputeFSI : public SoluteExchange::Visitor
{
    //TODO: use cudaEvent_t evuploaded;

    SolventWrap wsolvent;

    Logistic::KISS local_trunk;

    // moved global variables from fsi.cu
    cudaTextureObject_t texSolventParticles;
    cudaTextureObject_t texCellsStart;
    /* currently unused
    cudaTextureObject_t texCellsCount;*/
    int* packstarts_padded, *packcount;
    Particle** packstates;
    Acceleration** packresults;
    int hrecvpackcount[26], hrecvpackstarts_padded[27];
    const Particle* hrecvpackstates[26];
    Acceleration* hpackresults[26];

    bool firsttime;


public:

    void bind_solvent(SolventWrap wrap) { wsolvent = wrap; }

    ComputeFSI(MPI_Comm comm);
    ~ComputeFSI();

    void bulk(std::vector<ParticlesWrap> wsolutes, cudaStream_t stream);

    /*override of SoluteExchange::Visitor::halo*/
    void halo(ParticlesWrap solutes[26], cudaStream_t stream);
    void setup(const Particle * const solvent, const int npsolvent, const int * const cellsstart, const int * const cellscount);
};
