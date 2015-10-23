#pragma once

#include "common.h"
#include "migration.h"

template <typename T>
class MigratableBufferBase
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
        free_elements(data);
	    data = NULL;
        update();
	}

    void resize(const int n)
	{
	    assert(n >= 0);

	    size = n;
	    if (capacity >= n)
		    return;

        free_elements(data);

	    const int conservative_estimate = (int)ceil(1.1 * n);
	    capacity = 128 * ((conservative_estimate + 129) / 128);
        assert(capacity >= n);

        allocate_elements(&data, capacity);

#ifndef NDEBUG
        set_zero(data, capacity);
#endif
        update();
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
            free_elements(old);
	    }
        update();
	}
protected:
    virtual void allocate_elements(T** ptr, int count) = 0;
    virtual void copy_elements(T* dst, T* src, int count) = 0;
    virtual void set_zero(T* ptr, int count) = 0;
    virtual void free_elements(T* ptr) = 0;
    virtual void update() {}
};

template <typename T>
class MigratableDeviceBuffer : public MigratableBufferBase<T>, public Migratable<2>
{
public:
    MigratableDeviceBuffer(int n = 0) : MigratableBufferBase<T>(n) {}
protected:
    void allocate_elements(T** ptr, int count)
    {
        malloc_migratable_device((void**) ptr, count * sizeof(T));
    }

    void copy_elements(T* dst, T* src, int count)
    {
        CUDA_CHECK(cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyDeviceToDevice));
    }

    void set_zero(T* ptr, int count)
    {
        CUDA_CHECK(cudaMemset(ptr, 0, count * sizeof(T)));
    }

    void free_elements(T* ptr)
    {
        free_migratable(ptr);
    }
};

template <typename T>
class MigratableHostBuffer : public MigratableBufferBase<T>, public Migratable<2>
{
public:
    MigratableHostBuffer(int n = 0) : MigratableBufferBase<T>(n) {}
protected:
    void allocate_elements(T** ptr, int count)
    {
        malloc_migratable_host((void**) ptr, count * sizeof(T));
    }

    void copy_elements(T* dst, T* src, int count)
    {
        memcpy(dst, src, count * sizeof(T));
    }

    void set_zero(T* ptr, int count)
    {
        memset(ptr, 0, count * sizeof(T));
    }

    void free_elements(T* ptr)
    {
        free_migratable(ptr);
    }
};

template <typename T>
class MigratablePinnedBuffer : public MigratableBufferBase<T>, public Migratable<2>
{
public:
    T* devptr;

    MigratablePinnedBuffer(int n = 0) : MigratableBufferBase<T>(n) {}
protected:
    void allocate_elements(T** ptr, int count)
    {
        malloc_migratable_pinned((void**) ptr, count * sizeof(T));
    }

    void copy_elements(T* dst, T* src, int count)
    {
        CUDA_CHECK(cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyHostToHost));
    }

    void set_zero(T* ptr, int count)
    {
        CUDA_CHECK(cudaMemset(ptr, 0, count * sizeof(T)));
    }

    void free_elements(T* ptr)
    {
        free_migratable(ptr);
    }

    void update()
    {
        if (MigratableBufferBase<T>::data == NULL)
            devptr = NULL;
        else
            CUDA_CHECK(cudaHostGetDevicePointer(&devptr, MigratableBufferBase<T>::data, 0));
    }
};

template <typename T>
class MigratableDeviceBuffer2 : public MigratableBufferBase<T>
{
public:
    AnyMigratable* migratable;

    MigratableDeviceBuffer2(AnyMigratable* migratable = NULL, int n = 0):
        MigratableBufferBase<T>(n), migratable(migratable) {}
protected:
    void allocate_elements(T** ptr, int count)
    {
        assert(migratable);
        migratable->malloc_migratable_device((void**) ptr, count * sizeof(T));
    }

    void copy_elements(T* dst, T* src, int count)
    {
        CUDA_CHECK(cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyDeviceToDevice));
    }

    void set_zero(T* ptr, int count)
    {
        CUDA_CHECK(cudaMemset(ptr, 0, count * sizeof(T)));
    }

    void free_elements(T* ptr)
    {
        assert(migratable);
        migratable->free_migratable(ptr);
    }
};

template <typename T>
class MigratableHostBuffer2 : public MigratableBufferBase<T>
{
public:
    AnyMigratable* migratable;

    MigratableHostBuffer2(AnyMigratable* migratable = NULL, int n = 0):
        MigratableBufferBase<T>(n), migratable(migratable) {}
protected:
    void allocate_elements(T** ptr, int count)
    {
        assert(migratable);
        migratable->malloc_migratable_host((void**) ptr, count * sizeof(T));
    }

    void copy_elements(T* dst, T* src, int count)
    {
        memcpy(dst, src, count * sizeof(T));
    }

    void set_zero(T* ptr, int count)
    {
        memset(ptr, 0, count * sizeof(T));
    }

    void free_elements(T* ptr)
    {
        assert(migratable);
        migratable->free_migratable(ptr);
    }
};

template <typename T>
class MigratablePinnedBuffer2 : public MigratableBufferBase<T>
{
public:
    AnyMigratable* migratable;
    T* devptr;

    MigratablePinnedBuffer2(AnyMigratable* migratable = NULL, int n = 0):
        MigratableBufferBase<T>(n), migratable(migratable) {}
protected:
    void allocate_elements(T** ptr, int count)
    {
        assert(migratable);
        migratable->malloc_migratable_pinned((void**) ptr, count * sizeof(T));
    }

    void copy_elements(T* dst, T* src, int count)
    {
        CUDA_CHECK(cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyHostToHost));
    }

    void set_zero(T* ptr, int count)
    {
        CUDA_CHECK(cudaMemset(ptr, 0, count * sizeof(T)));
    }

    void free_elements(T* ptr)
    {
        assert(migratable);
        migratable->free_migratable(ptr);
    }

    void update()
    {
        if (MigratableBufferBase<T>::data == NULL)
            devptr = NULL;
        else
            CUDA_CHECK(cudaHostGetDevicePointer(&devptr, MigratableBufferBase<T>::data, 0));
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
