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

    void mallocMigratableHost(void** ptr, int size);
    void mallocMigratableDevice(void** ptr, int size);
    void freeMigratable(void* ptr);

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

    Buffer& getInactiveBuffer()
    {
        for (int i = 0; i < MAX_BUFFERS; ++i)
            if (buffers[i].status == Buffer::INACTIVE)
                return buffers[i];
        assert(false);
        return buffers[0];
    }

    void setInactiveBuffer(void** ptr, int size, Buffer::Status status)
    {
        Buffer& b = getInactiveBuffer();
        b.ptr = *ptr;
        b.size = size;
        b.offset = getPtrOffset(ptr);
        b.status = status;
    }

    int getPtrOffset(void** ptr) const
    {
        return (int) ((ptrdiff_t) ptr - (ptrdiff_t) this);
    }

    Buffer buffers[MAX_BUFFERS];
};
