#pragma once

#include <cassert>
#include <cstddef>

class Migratable
{
public:
    enum {
        MAX_BUFFERS = 16  
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

    void allocate_inactive_buffer(void** ptr, int size, Buffer::Status status)
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

    static void allocate(void** ptr, int size, Buffer::Status status);
    static void free(void* ptr, Buffer::Status status);

    Buffer buffers[MAX_BUFFERS];
};
