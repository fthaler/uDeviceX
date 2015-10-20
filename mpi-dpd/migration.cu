#include <iostream>

#include <mpi.h>

#include "common.h"
#include "migration.h"

Migratable::Migratable()
{
    memset(buffers, 0, sizeof(buffers));
#ifdef AMPI
    MPI_Register(this, pup);
#endif
}

void Migratable::mallocMigratableHost(void** ptr, int size)
{
    *ptr = malloc(size);
    setInactiveBuffer(ptr, size, Buffer::HOST);
}

void Migratable::mallocMigratableDevice(void** ptr, int size)
{
    CUDA_CHECK(cudaMalloc(ptr, size));
    setInactiveBuffer(ptr, size, Buffer::DEVICE);
}

void Migratable::freeMigratable(void* ptr)
{
    if (ptr == NULL)
        return;
    for (int i = 0; i < MAX_BUFFERS; ++i) {
        Buffer& b = buffers[i];
        if (b.status != Buffer::INACTIVE && b.ptr == ptr) {
            switch (b.status) {
            case Buffer::HOST:
                free(ptr);
                break;
            case Buffer::DEVICE:
                CUDA_CHECK(cudaFree(ptr));
                break;
            }
            b.status = Buffer::INACTIVE;
            return;
        }
    }
}

void Migratable::pup(pup_er p, void *d)
{
    Migratable* m = (Migratable*) d;

    if (pup_isUnpacking(p))
    {
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            Buffer& b = m->buffers[i];
            if (b.status != Buffer::INACTIVE)
                b.ptr = malloc(b.size);
        }
    }
    if (pup_isPacking(p))
    {
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            Buffer& b = m->buffers[i];
            if (b.status == Buffer::DEVICE) {
                void* newPtr = malloc(b.size);
                CUDA_CHECK(cudaMemcpy(newPtr, b.ptr, b.size, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaFree(b.ptr));
                b.ptr = newPtr;
            }
        }
    }
    for (int i = 0; i < MAX_BUFFERS; ++i) {
        Buffer& b = m->buffers[i];
        if (b.status != Buffer::INACTIVE)
            pup_bytes(p, b.ptr, b.size);
    }
    if (pup_isUnpacking(p))
    {
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            Buffer& b = m->buffers[i];
            if (b.status != Buffer::INACTIVE) {
                if (b.status == Buffer::DEVICE) {
                    void* newPtr;
                    CUDA_CHECK(cudaMalloc(&newPtr, b.size));
                    CUDA_CHECK(cudaMemcpy(newPtr, b.ptr, b.size, cudaMemcpyHostToDevice));
                    b.ptr = newPtr;
                }
                void** memberPtr = (void**) ((ptrdiff_t) m + b.offset);
                *memberPtr = b.ptr;
            }
        }
    }
    if (pup_isPacking(p))
    {
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            Buffer& b = m->buffers[i];
            if (b.status != Buffer::INACTIVE)
                free(b.ptr);
        }
    }
}
