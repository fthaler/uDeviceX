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

void Migratable::malloc_migratable(void** ptr, int size)
{
    allocate_inactive_buffer(ptr, size, Buffer::MALLOCED);
}

void Migratable::malloc_migratable_device(void** ptr, int size)
{
    allocate_inactive_buffer(ptr, size, Buffer::CUDA_DEVICE);
}

void Migratable::malloc_migratable_host(void** ptr, int size)
{
    allocate_inactive_buffer(ptr, size, Buffer::CUDA_HOST);
}

void Migratable::malloc_migratable_pinned(void** ptr, int size)
{
    allocate_inactive_buffer(ptr, size, Buffer::CUDA_PINNED);
}

void Migratable::free_migratable(void* ptr)
{
    if (ptr == NULL)
        return;
    for (int i = 0; i < MAX_BUFFERS; ++i) {
        Buffer& b = buffers[i];
        if (b.status != Buffer::INACTIVE && b.ptr == ptr) {
            free(b.ptr, b.status);
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
            Buffer::Status alloc_status = b.status;
            if (alloc_status == Buffer::CUDA_DEVICE)
                alloc_status = Buffer::MALLOCED;
            allocate(&b.ptr, b.size, alloc_status);
        }
    }
    if (pup_isPacking(p))
    {
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            Buffer& b = m->buffers[i];
            if (b.status == Buffer::CUDA_DEVICE) {
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
                if (b.status == Buffer::CUDA_DEVICE) {
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
            Buffer::Status alloc_status = b.status;
            if (alloc_status == Buffer::CUDA_DEVICE)
                alloc_status = Buffer::MALLOCED;
            free(b.ptr, alloc_status);
            b.ptr = NULL;
        }
    }
}

void Migratable::allocate(void** ptr, int size, Buffer::Status status)
{
    switch (status) {
    case Buffer::INACTIVE:
        *ptr = NULL;
        break;
    case Buffer::MALLOCED:
        *ptr = malloc(size);
        break;
    case Buffer::CUDA_DEVICE:
        CUDA_CHECK(cudaMalloc(ptr, size));
        break;
    case Buffer::CUDA_HOST:
        CUDA_CHECK(cudaHostAlloc(ptr, size, cudaHostAllocDefault));
        break;
    case Buffer::CUDA_PINNED:
        CUDA_CHECK(cudaHostAlloc(ptr, size, cudaHostAllocMapped));
        break;
    }
}

void Migratable::free(void* ptr, Buffer::Status status)
{
    if (ptr == NULL)
        return;
    switch (status) {
    case Buffer::INACTIVE:
        break;
    case Buffer::MALLOCED:
        ::free(ptr);
        break;
    case Buffer::CUDA_DEVICE:
        CUDA_CHECK(cudaFree(ptr));
        break;
    case Buffer::CUDA_HOST:
    case Buffer::CUDA_PINNED:
        CUDA_CHECK(cudaFreeHost(ptr));
        break;
    }
}
