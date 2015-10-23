#pragma once

#include <cassert>
#include <cstddef>

#include <mpi.h>

#include "common.h"

template <int B>
class Migratable
{
public:
    enum {
        MAX_BUFFERS = B
    };

protected:
    Migratable();

    void malloc_migratable(void** ptr, int size);
    void malloc_migratable_host(void** ptr, int size);
    void malloc_migratable_device(void** ptr, int size);
    void malloc_migratable_pinned(void** ptr, int size);
    void free_migratable(void* ptr);

private:
    struct Buffer
    {
        enum Status
        {
            INACTIVE = 0,
            MALLOCED,
            CUDA_DEVICE,
            CUDA_HOST,
            CUDA_PINNED
        };

        void *ptr;
        int size;
        int offset;
        Status status;
    };

    static void pup(pup_er p, void *d);

    Buffer& get_inactive_buffer()
    {
        for (int i = 0; i < MAX_BUFFERS; ++i)
            if (buffers[i].status == Buffer::INACTIVE)
                return buffers[i];
        assert(false);
        return buffers[0];
    }

    void allocate_inactive_buffer(void** ptr, int size, typename Buffer::Status status)
    {
        Buffer& b = get_inactive_buffer();
        allocate(ptr, size, status);
        b.ptr = *ptr;
        b.size = size;
        b.offset = get_ptr_offset(ptr);
        b.status = status;
    }

    int get_ptr_offset(void** ptr) const
    {
        return (int) ((ptrdiff_t) ptr - (ptrdiff_t) this);
    }

    static void allocate(void** ptr, int size, typename Buffer::Status status);
    static void free(void* ptr, typename Buffer::Status status);

    Buffer buffers[MAX_BUFFERS];
};

template <int B>
Migratable<B>::Migratable()
{
    memset(buffers, 0, sizeof(buffers));
#ifdef AMPI
    MPI_Register(this, pup);
#endif
}

template <int B>
void Migratable<B>::malloc_migratable(void** ptr, int size)
{
    allocate_inactive_buffer(ptr, size, Buffer::MALLOCED);
}

template <int B>
void Migratable<B>::malloc_migratable_device(void** ptr, int size)
{
    allocate_inactive_buffer(ptr, size, Buffer::CUDA_DEVICE);
}

template <int B>
void Migratable<B>::malloc_migratable_host(void** ptr, int size)
{
    allocate_inactive_buffer(ptr, size, Buffer::CUDA_HOST);
}

template <int B>
void Migratable<B>::malloc_migratable_pinned(void** ptr, int size)
{
    allocate_inactive_buffer(ptr, size, Buffer::CUDA_PINNED);
}

template <int B>
void Migratable<B>::free_migratable(void* ptr)
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

template <int B>
void Migratable<B>::pup(pup_er p, void *d)
{
    Migratable* m = (Migratable*) d;

    if (pup_isUnpacking(p))
    {
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            Buffer& b = m->buffers[i];
            typename Buffer::Status alloc_status = b.status;
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
            typename Buffer::Status alloc_status = b.status;
            if (alloc_status == Buffer::CUDA_DEVICE)
                alloc_status = Buffer::MALLOCED;
            free(b.ptr, alloc_status);
            b.ptr = NULL;
        }
    }
}

template <int B>
void Migratable<B>::allocate(void** ptr, int size, typename Buffer::Status status)
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

template <int B>
void Migratable<B>::free(void* ptr, typename Buffer::Status status)
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
