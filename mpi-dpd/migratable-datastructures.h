#pragma once

#include "common.h"
#include "migration.h"

template <typename T>
class MigratableBufferBase : public Migratable<2>
{
public:
    int capacity, size;
    T * data;

    MigratableBufferBase(int n = 0): capacity(0), size(0), data(NULL) { resize(n);}

    virtual ~MigratableBufferBase()
	{
        dispose();
	}

    void dispose()
	{
        free_migratable(data);
	    data = NULL;
	}

    void resize(const int n)
	{
	    assert(n >= 0);

	    size = n;
	    if (capacity >= n)
		    return;

        free_migratable(data);

	    const int conservative_estimate = (int)ceil(1.1 * n);
	    capacity = 128 * ((conservative_estimate + 129) / 128);

        allocate_elements(&data, capacity);

#ifndef NDEBUG
        set_zero(data, capacity);
#endif
	}

    void preserve_resize(const int n)
	{
	    assert(n >= 0);

	    T * old = data;

	    const int oldsize = size;

	    size = n;
	    if (capacity >= n)
		    return;

	    const int conservative_estimate = (int)ceil(1.1 * n);
	    capacity = 128 * ((conservative_estimate + 129) / 128);

        allocate_elements(&data, capacity);

	    if (old != NULL)
	    {
            copy_elements(data, old, oldsize);
            free_migratable(old);
	    }
	}
protected:
    virtual void allocate_elements(T** ptr, int count) = 0;
    virtual void copy_elements(T* dst, T* src, int count) = 0;
    virtual void set_zero(T* ptr, int count) = 0;
};

template <typename T>
class MigratableDeviceBuffer : public MigratableBufferBase<T>
{
protected:
    void allocate_elements(T** ptr, int count)
    {
        Migratable<2>::malloc_migratable_device((void**) ptr, count * sizeof(T));
    }

    void copy_elements(T* dst, T* src, int count)
    {
        CUDA_CHECK(cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyDeviceToDevice));
    }

    void set_zero(T* ptr, int count)
    {
        CUDA_CHECK(cudaMemset(ptr, 0, count * sizeof(T)));
    }
};

template <typename T>
class MigratableHostBuffer : public MigratableBufferBase<T>
{
protected:
    void allocate_elements(T** ptr, int count)
    {
        Migratable<2>::malloc_migratable_host((void**) ptr, count * sizeof(T));
    }

    void copy_elements(T* dst, T* src, int count)
    {
        memcpy(dst, src, count * sizeof(T));
    }

    void set_zero(T* ptr, int count)
    {
        memset(ptr, 0, count * sizeof(T));
    }
};

template <typename T>
class MigratablePinnedBuffer : public MigratableBufferBase<T>
{
protected:
    void allocate_elements(T** ptr, int count)
    {
        Migratable<2>::malloc_migratable_pinned((void**) ptr, count * sizeof(T));
    }

    void copy_elements(T* dst, T* src, int count)
    {
        memcpy(dst, src, count * sizeof(T));
    }

    void set_zero(T* ptr, int count)
    {
        memset(ptr, 0, count * sizeof(T));
    }
};

class MigratableCellLists : public Migratable<2>, public CellListsBase
{
public:
    MigratableCellLists(Globals* globals, const int LX, const int LY, const int LZ)
        : CellListsBase(globals, LX, LY, LZ)
    {
        malloc_migratable_device((void**) &start, sizeof(int) * ncells);
        malloc_migratable_device((void**) &count, sizeof(int) * ncells);
    }

    ~MigratableCellLists()
    {
        free_migratable(start);
        free_migratable(count);
    }
};
