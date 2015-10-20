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

    void malloc_migratable_host(void** ptr, int size);
    void malloc_migratable_device(void** ptr, int size);
    void free_migratable(void* ptr);

private:
    struct Buffer
    {
        enum Status
        {
            INACTIVE = 0,
            HOST,
            DEVICE
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

    void set_inactive_buffer(void** ptr, int size, Buffer::Status status)
    {
        Buffer& b = get_inactive_buffer();
        b.ptr = *ptr;
        b.size = size;
        b.offset = get_ptr_offset(ptr);
        b.status = status;
    }

    int get_ptr_offset(void** ptr) const
    {
        return (int) ((ptrdiff_t) ptr - (ptrdiff_t) this);
    }

    Buffer buffers[MAX_BUFFERS];
};
