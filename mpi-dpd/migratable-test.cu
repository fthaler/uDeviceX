#include <cassert>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <mpi.h>

#include "migration.h"

class MigratableThing : public Migratable
{
public:
    explicit MigratableThing() : nPtrs(0), bufferSize(512), id(id) {}

    void allocHostData()
    {
        if (nPtrs == MAX_BUFFERS)
            return;
        mallocMigratableHost(&ptrs[nPtrs], bufferSize);
        isOnDevice[nPtrs++] = false;
    }

    void allocDeviceData()
    {
        if (nPtrs == MAX_BUFFERS)
            return;
        mallocMigratableDevice(&ptrs[nPtrs], bufferSize);
        isOnDevice[nPtrs++] = true;
    }

    void doSomeStuff()
    {
        for (int i = 0; i < nPtrs; ++i) {
            if (isOnDevice[i])
                cudaMemset(ptrs[i], 0, bufferSize);
            else
                memset(ptrs[i], 0, bufferSize);
        }
    }

    void freeLast()
    {
        if (nPtrs == 0)
            return;
        freeMigratable(ptrs[--nPtrs]);
    }

    void setBufferSize(int newBufferSize) { bufferSize = newBufferSize; }
    void setId(int newId) { id = newId; }
    int getId() const { return id; }
private:
    void* ptrs[MAX_BUFFERS];
    bool isOnDevice[MAX_BUFFERS];
    int nPtrs, bufferSize, id;
};

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    srand(2342347 * rank);

    const int n = 7;
    MigratableThing things[n];
    for (int i = 0; i < n; ++i) {
        things[i].setBufferSize(rand() % 1024);
        things[i].setId(i + rank * n);
    }

    for (int i = 0; i < 100; ++i)  {
        char processorName[MPI_MAX_PROCESSOR_NAME];
        int processorNameLength;
        MPI_Get_processor_name(processorName, &processorNameLength);
        processorName[processorNameLength + 1] = 0;
        std::cout << processorName << ": ";

        int j = rand() % n;
        MigratableThing& mt = things[j];
        std::cout << "things[" << mt.getId() << "]";
        switch (rand() % 3) {
        case 0:
            mt.allocHostData();
            std::cout << ".allocHostData()";
            break;
        case 1:
            mt.allocDeviceData();
            std::cout << ".allocDeviceData()";
            break;
        case 2:
            mt.freeLast();
            std::cout << ".freeLast()";
            break;
        }
        std::cout << std::endl;

        for (int j = 0; j < n; ++j)
            things[j].doSomeStuff();

#ifdef AMPI
        MPI_Migrate();
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0)
            std::cout << "--- MIGRATION ---" << std::endl;
        MPI_Barrier(MPI_COMM_WORLD);
#endif
    }
    

    MPI_Finalize();
    return 0;
};
