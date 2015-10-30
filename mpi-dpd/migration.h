#pragma once

#include <cassert>
#include <cstddef>

#include <mpi.h>

#include "common.h"

class AnyMigratable
{
public:
    virtual void malloc_migratable(void** ptr, int size) = 0;
    virtual void malloc_migratable_host(void** ptr, int size) = 0;
    virtual void malloc_migratable_device(void** ptr, int size) = 0;
    virtual void malloc_migratable_pinned(void** ptr, int size) = 0;
    virtual void free_migratable(void* ptr) = 0;
    virtual void create_migratable_event(cudaEvent_t* event, unsigned flags = 0) = 0;
    virtual void destroy_migratable_event(cudaEvent_t event) = 0;
};

template <int B, int E = 0>
class Migratable : public AnyMigratable
{
public:
    enum {
        MAX_BUFFERS = B
    };
    enum {
        MAX_EVENTS = E
    };

    Migratable();

    void malloc_migratable(void** ptr, int size);
    void malloc_migratable_host(void** ptr, int size);
    void malloc_migratable_device(void** ptr, int size);
    void malloc_migratable_pinned(void** ptr, int size);
    void free_migratable(void* ptr);
    void create_migratable_event(cudaEvent_t* event, unsigned flags = 0);
    void destroy_migratable_event(cudaEvent_t event);

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

    struct Event
    {
        enum Status
        {
            INACTIVE = 0,
            ACTIVE
        };

        cudaEvent_t event;
        unsigned flags;
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

    Event& get_inactive_event()
    {
        for (int i = 0; i < MAX_EVENTS; ++i)
            if (events[i].status == Event::INACTIVE)
                return events[i];
        assert(false);
        return events[0];
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
    Event events[MAX_EVENTS];
};

template <int B, int E>
Migratable<B, E>::Migratable()
{
    memset(buffers, 0, sizeof(buffers));
#ifdef AMPI
    MPI_Register(this, pup);
#endif
}

template <int B, int E>
void Migratable<B, E>::malloc_migratable(void** ptr, int size)
{
    allocate_inactive_buffer(ptr, size, Buffer::MALLOCED);
}

template <int B, int E>
void Migratable<B, E>::malloc_migratable_device(void** ptr, int size)
{
    allocate_inactive_buffer(ptr, size, Buffer::CUDA_DEVICE);
}

template <int B, int E>
void Migratable<B, E>::malloc_migratable_host(void** ptr, int size)
{
    allocate_inactive_buffer(ptr, size, Buffer::CUDA_HOST);
}

template <int B, int E>
void Migratable<B, E>::malloc_migratable_pinned(void** ptr, int size)
{
    allocate_inactive_buffer(ptr, size, Buffer::CUDA_PINNED);
}

template <int B, int E>
void Migratable<B, E>::free_migratable(void* ptr)
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
    assert(false);
}

template <int B, int E>
void Migratable<B, E>::create_migratable_event(cudaEvent_t* event, unsigned flags)
{
    Event& e = get_inactive_event();
    CUDA_CHECK(cudaEventCreateWithFlags(event, flags));
    e.event = *event;
    e.flags = flags;
    e.offset = get_ptr_offset((void**) event);
    e.status = Event::ACTIVE;
}

template <int B, int E>
void Migratable<B, E>::destroy_migratable_event(cudaEvent_t event)
{
    for (int i = 0; i < MAX_EVENTS; ++i) {
        Event& e = events[i];
        if (e.status == Event::ACTIVE && e.event == event) {
            CUDA_CHECK(cudaEventDestroy(event));
            e.status = Event::INACTIVE;
            return;
        }
    }
    assert(false);
}

template <int B, int E>
void Migratable<B, E>::pup(pup_er p, void *d)
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

        for (int i = 0; i < MAX_EVENTS; ++i) {
            Event& e = m->events[i];
            if (e.status == Event::ACTIVE)
                CUDA_CHECK(cudaEventDestroy(e.event));
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

        for (int i = 0; i < MAX_EVENTS; ++i) {
            Event& e = m->events[i];
            if (e.status == Event::ACTIVE) {
                CUDA_CHECK(cudaEventCreateWithFlags(&e.event, e.flags));
                cudaEvent_t* memberPtr = (cudaEvent_t*) ((ptrdiff_t) m + e.offset);
                *memberPtr = e.event;
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

template <int B, int E>
void Migratable<B, E>::allocate(void** ptr, int size, typename Buffer::Status status)
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

template <int B, int E>
void Migratable<B, E>::free(void* ptr, typename Buffer::Status status)
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
