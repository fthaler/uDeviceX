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

template <int B, int E = 0, int A = 0>
class Migratable : public AnyMigratable
{
public:
    enum {
        MAX_BUFFERS = B
    };
    enum {
        MAX_EVENTS = E
    };
    enum {
        MAX_ARRAYS = A
    };

    Migratable();

    void malloc_migratable(void** ptr, int size);
    void malloc_migratable_host(void** ptr, int size);
    void malloc_migratable_device(void** ptr, int size);
    void malloc_migratable_pinned(void** ptr, int size);
    void free_migratable(void* ptr);

    void malloc_migratable_array(cudaArray_t* array, cudaChannelFormatDesc* desc,
                                 cudaExtent extent, unsigned flags = 0);
    void free_migratable_array(cudaArray_t array);

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

    struct Array
    {
        enum Status
        {
            INACTIVE = 0,
            ACTIVE
        };

        cudaArray_t array;
        cudaChannelFormatDesc desc;
        cudaExtent extent;
        unsigned flags;
        int offset;
        Status status;
        void *hostptr;
    };

#ifdef AMPI
    static void pup(pup_er p, void *d);
#endif

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

    Array& get_inactive_array()
    {
        for (int i = 0; i < MAX_ARRAYS; ++i)
            if (arrays[i].status == Array::INACTIVE)
                return arrays[i];
        assert(false);
        return arrays[0];
    }

    void allocate_inactive_buffer(void** ptr, int size, typename Buffer::Status status)
    {
        Buffer& b = get_inactive_buffer();
        allocate_buffer(ptr, size, status);
        b.ptr = *ptr;
        b.size = size;
        b.offset = get_ptr_offset(ptr);
        b.status = status;
    }

    int get_ptr_offset(void** ptr) const
    {
        return (int) ((ptrdiff_t) ptr - (ptrdiff_t) this);
    }

    static void allocate_buffer(void** ptr, int size, typename Buffer::Status status);
    static void free_buffer(void* ptr, typename Buffer::Status status);

    Buffer buffers[MAX_BUFFERS];
    Event events[MAX_EVENTS];
    Array arrays[MAX_ARRAYS];
};

template <int B, int E, int A>
Migratable<B, E, A>::Migratable()
{
    memset(buffers, 0, sizeof(buffers));
    memset(events, 0, sizeof(events));
    memset(arrays, 0, sizeof(arrays));
#ifdef AMPI
    MPI_Register(this, pup);
#endif
}

template <int B, int E, int A>
void Migratable<B, E, A>::malloc_migratable(void** ptr, int size)
{
    allocate_inactive_buffer(ptr, size, Buffer::MALLOCED);
}

template <int B, int E, int A>
void Migratable<B, E, A>::malloc_migratable_device(void** ptr, int size)
{
    allocate_inactive_buffer(ptr, size, Buffer::CUDA_DEVICE);
}

template <int B, int E, int A>
void Migratable<B, E, A>::malloc_migratable_host(void** ptr, int size)
{
    allocate_inactive_buffer(ptr, size, Buffer::CUDA_HOST);
}

template <int B, int E, int A>
void Migratable<B, E, A>::malloc_migratable_pinned(void** ptr, int size)
{
    allocate_inactive_buffer(ptr, size, Buffer::CUDA_PINNED);
}

template <int B, int E, int A>
void Migratable<B, E, A>::free_migratable(void* ptr)
{
    if (ptr == NULL)
        return;
    for (int i = 0; i < MAX_BUFFERS; ++i) {
        Buffer& b = buffers[i];
        if (b.status != Buffer::INACTIVE && b.ptr == ptr) {
            free_buffer(b.ptr, b.status);
            b.status = Buffer::INACTIVE;
            return;
        }
    }
    assert(false);
}

template <int B, int E, int A>
void Migratable<B, E, A>::malloc_migratable_array(cudaArray_t* array, cudaChannelFormatDesc* desc, cudaExtent extent, unsigned flags)
{
    Array& a = get_inactive_array();
    CUDA_CHECK(cudaMalloc3DArray(array, desc, extent, flags));
    a.array = *array;
    a.desc = *desc;
    a.extent = extent;
    a.flags = flags;
    a.offset = get_ptr_offset((void**) array);
    a.status = Array::ACTIVE;
    a.hostptr = NULL;
}

template <int B, int E, int A>
void Migratable<B, E, A>::free_migratable_array(cudaArray_t array)
{
    for (int i = 0; i < MAX_ARRAYS; ++i) {
        Array& a = arrays[i];
        if (a.status == Array::ACTIVE && a.array == array) {
            CUDA_CHECK(cudaFreeArray(array));
            a.status = Array::INACTIVE;
            return;
        }
    }
    assert(false);
}

template <int B, int E, int A>
void Migratable<B, E, A>::create_migratable_event(cudaEvent_t* event, unsigned flags)
{
    Event& e = get_inactive_event();
    CUDA_CHECK(cudaEventCreateWithFlags(event, flags));
    e.event = *event;
    e.flags = flags;
    e.offset = get_ptr_offset((void**) event);
    e.status = Event::ACTIVE;
}

template <int B, int E, int A>
void Migratable<B, E, A>::destroy_migratable_event(cudaEvent_t event)
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

#ifdef AMPI
template <int B, int E, int A>
void Migratable<B, E, A>::pup(pup_er p, void *d)
{
    Migratable* m = (Migratable*) d;

    if (pup_isUnpacking(p))
    {
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            Buffer& b = m->buffers[i];
            typename Buffer::Status alloc_status = b.status;
            if (alloc_status == Buffer::CUDA_DEVICE)
                alloc_status = Buffer::MALLOCED;
            allocate_buffer(&b.ptr, b.size, alloc_status);
        }

        for (int i = 0; i < MAX_ARRAYS; ++i) {
            Array& a = m->arrays[i];
            if (a.status == Array::ACTIVE) {
                int bytes_per_entry = a.desc.w + a.desc.x + a.desc.y + a.desc.z;
                size_t entries = a.extent.depth * a.extent.height * a.extent.width;
                a.hostptr = malloc(entries * bytes_per_entry);
            }

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

        for (int i = 0; i < MAX_ARRAYS; ++i) {
            Array& a = m->arrays[i];
            if (a.status == Array::ACTIVE) {
                int bytes_per_entry = a.desc.w + a.desc.x + a.desc.y + a.desc.z;
                size_t entries = a.extent.depth * a.extent.height * a.extent.width;
                a.hostptr = malloc(entries * bytes_per_entry);
                cudaMemcpy3DParms p = {0};
                p.srcArray = a.array;
                p.dstPtr = make_cudaPitchedPtr(a.hostptr, a.extent.depth * bytes_per_entry,
                                               a.extent.depth, a.extent.height);
                p.extent = a.extent;
                p.kind = cudaMemcpyDeviceToHost;
                CUDA_CHECK(cudaMemcpy3D(&p));
                CUDA_CHECK(cudaFreeArray(a.array));
            }

        }

        for (int i = 0; i < MAX_EVENTS; ++i) {
            Event& e = m->events[i];
            if (e.status == Event::ACTIVE) {
                CUDA_CHECK(cudaEventSynchronize(e.event));
                CUDA_CHECK(cudaEventDestroy(e.event));
            }
        }
    }
    for (int i = 0; i < MAX_BUFFERS; ++i) {
        Buffer& b = m->buffers[i];
        if (b.status != Buffer::INACTIVE)
            pup_bytes(p, b.ptr, b.size);
    }
    for (int i = 0; i < MAX_ARRAYS; ++i) {
        Array& a = m->arrays[i];
        if (a.status != Array::INACTIVE) {
            int bytes_per_entry = a.desc.w + a.desc.x + a.desc.y + a.desc.z;
            size_t entries = a.extent.depth * a.extent.height * a.extent.width;
            pup_bytes(p, a.hostptr, entries * bytes_per_entry);
        }
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
                    free(b.ptr);
                    b.ptr = newPtr;
                }
                void** memberPtr = (void**) ((ptrdiff_t) m + b.offset);
                *memberPtr = b.ptr;
            }
        }

        for (int i = 0; i < MAX_ARRAYS; ++i) {
            Array& a = m->arrays[i];
            if (a.status == Array::ACTIVE) {
                CUDA_CHECK(cudaMalloc3DArray(&a.array, &a.desc, a.extent));

                cudaMemcpy3DParms p = {0};
                int bytes_per_entry = a.desc.w + a.desc.x + a.desc.y + a.desc.z;
                p.srcPtr = make_cudaPitchedPtr(a.hostptr, a.extent.depth * bytes_per_entry,
                                               a.extent.depth, a.extent.height);
                p.dstArray = a.array;
                p.extent = a.extent;
                p.kind = cudaMemcpyHostToDevice;
                CUDA_CHECK(cudaMemcpy3D(&p));
                free(a.hostptr);
                a.hostptr = NULL;
                
                cudaArray_t* memberPtr = (cudaArray_t*) ((ptrdiff_t) m + a.offset);
                *memberPtr = a.array;
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
            free_buffer(b.ptr, alloc_status);
            b.ptr = NULL;
        }
        for (int i = 0; i < MAX_ARRAYS; ++i) {
            Array& a = m->arrays[i];
            free(a.hostptr);
            a.hostptr = NULL;
        }
    }
}
#endif

template <int B, int E, int A>
void Migratable<B, E, A>::allocate_buffer(void** ptr, int size, typename Buffer::Status status)
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

template <int B, int E, int A>
void Migratable<B, E, A>::free_buffer(void* ptr, typename Buffer::Status status)
{
    if (ptr == NULL)
        return;
    switch (status) {
    case Buffer::INACTIVE:
        break;
    case Buffer::MALLOCED:
        free(ptr);
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
