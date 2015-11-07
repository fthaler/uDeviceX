#pragma once

#include "common.h"
#include "migration.h"

template <typename T, typename Derived>
class MigratableBufferBase
{
public:
    int capacity, size;
    T * data;

    MigratableBufferBase(int n = 0, bool noresize = false): capacity(0), size(0), data(NULL)
    {
        if (!noresize)
            resize(n);
    }

    virtual ~MigratableBufferBase()
	{
        dispose();
	}

    void dispose()
	{
        free_elements(data);
	    data = NULL;
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
	}
protected:
    void allocate_elements(T** ptr, int count)
    {
        static_cast<Derived*>(this)->allocate_elements_d(ptr, count);
    }
    void copy_elements(T* dst, T* src, int count)
    {
        static_cast<Derived*>(this)->copy_elements_d(dst, src, count);
    }
    void set_zero(T* ptr, int count)
    {
        static_cast<Derived*>(this)->set_zero_d(ptr, count);
    }
    void free_elements(T* ptr)
    {
        static_cast<Derived*>(this)->free_elements_d(ptr);
    }
};

template <typename T, typename Derived>
class MigratableBufferBase2 : public MigratableBufferBase<T, Derived>
{
public:
    MigratableBufferBase2(AnyMigratable* migratable, int n = 0)
        : MigratableBufferBase<T, Derived>(n, true),
        migratable_ptr_offset((int) ((ptrdiff_t) migratable - (ptrdiff_t) this)) {
            assert(migratable != NULL);
            MigratableBufferBase<T, Derived>::resize(n);
        }
    virtual ~MigratableBufferBase2() {}
protected:
    AnyMigratable* get_migratable() {
        AnyMigratable* migratable = (AnyMigratable*) ((ptrdiff_t) this + migratable_ptr_offset);
        assert(migratable != NULL);
        return migratable;
    }
private:
    int migratable_ptr_offset;
};


template <typename T>
class MigratableBuffer
    : public MigratableBufferBase<T, MigratableBuffer<T> >, public Migratable<2>
{
public:
    MigratableBuffer(int n = 0)
        : MigratableBufferBase<T, MigratableBuffer<T> >(n) {}

    void allocate_elements_d(T** ptr, int count)
    {
        Migratable<2>::malloc_migratable((void**) ptr, count * sizeof(T));
    }

    void copy_elements_d(T* dst, T* src, int count)
    {
        memcpy(dst, src, count * sizeof(T));
    }

    void set_zero_d(T* ptr, int count)
    {
        memset(ptr, 0, count * sizeof(T));
    }

    void free_elements_d(T* ptr)
    {
        Migratable<2>::free_migratable(ptr);
    }
};

template <typename T>
class MigratableDeviceBuffer
    : public MigratableBufferBase<T, MigratableDeviceBuffer<T> >, public Migratable<2>
{
public:
    MigratableDeviceBuffer(int n = 0)
        : MigratableBufferBase<T, MigratableDeviceBuffer<T> >(n) {}

    void allocate_elements_d(T** ptr, int count)
    {
        Migratable<2>::malloc_migratable_device((void**) ptr, count * sizeof(T));
    }

    void copy_elements_d(T* dst, T* src, int count)
    {
        CUDA_CHECK(cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyDeviceToDevice));
    }

    void set_zero_d(T* ptr, int count)
    {
        CUDA_CHECK(cudaMemset(ptr, 0, count * sizeof(T)));
    }

    void free_elements_d(T* ptr)
    {
        Migratable<2>::free_migratable(ptr);
    }
};

template <typename T>
class MigratableHostBuffer
    : public MigratableBufferBase<T, MigratableHostBuffer<T> >, public Migratable<2>
{
public:
    MigratableHostBuffer(int n = 0)
        : MigratableBufferBase<T, MigratableHostBuffer<T> >(n) {}

    void allocate_elements_d(T** ptr, int count)
    {
        Migratable<2>::malloc_migratable_host((void**) ptr, count * sizeof(T));
    }

    void copy_elements_d(T* dst, T* src, int count)
    {
        memcpy(dst, src, count * sizeof(T));
    }

    void set_zero_d(T* ptr, int count)
    {
        memset(ptr, 0, count * sizeof(T));
    }

    void free_elements_d(T* ptr)
    {
        Migratable<2>::free_migratable(ptr);
    }
};

template <typename T>
class MigratablePinnedBuffer
    : public MigratableBufferBase<T, MigratablePinnedBuffer<T> >, public Migratable<2>
{
public:
    MigratablePinnedBuffer(int n = 0)
        : MigratableBufferBase<T, MigratablePinnedBuffer<T> >(n) {}

    void allocate_elements_d(T** ptr, int count)
    {
        Migratable<2>::malloc_migratable_pinned((void**) ptr, count * sizeof(T));
    }

    void copy_elements_d(T* dst, T* src, int count)
    {
        CUDA_CHECK(cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyHostToHost));
    }

    void set_zero_d(T* ptr, int count)
    {
        CUDA_CHECK(cudaMemset(ptr, 0, count * sizeof(T)));
    }

    void free_elements_d(T* ptr)
    {
        Migratable<2>::free_migratable(ptr);
    }

    T* devptr()
    {
        if (MigratableBufferBase<T, MigratablePinnedBuffer<T> >::data == NULL)
            return NULL;
        T* ptr;
        CUDA_CHECK(cudaHostGetDevicePointer(&ptr,
            MigratableBufferBase<T, MigratablePinnedBuffer<T> >::data, 0));
        return ptr;
    }
};

template <typename T>
class MigratableBuffer2 : public MigratableBufferBase2<T, MigratableBuffer2<T> >
{
private:
    typedef MigratableBufferBase2<T, MigratableBuffer2<T> > Base;
public:
    MigratableBuffer2(AnyMigratable* migratable, int n = 0):
        Base(migratable, n) {}

    void allocate_elements_d(T** ptr, int count)
    {
        Base::get_migratable()->malloc_migratable((void**) ptr, count * sizeof(T));
    }

    void copy_elements_d(T* dst, T* src, int count)
    {
        memcpy(dst, src, count * sizeof(T));
    }

    void set_zero_d(T* ptr, int count)
    {
        memset(ptr, 0, count * sizeof(T));
    }

    void free_elements_d(T* ptr)
    {
        Base::get_migratable()->free_migratable(ptr);
    }
};

template <typename T>
class MigratableDeviceBuffer2 : public MigratableBufferBase2<T, MigratableDeviceBuffer2<T> >
{
private:
    typedef MigratableBufferBase2<T, MigratableDeviceBuffer2<T> > Base;
public:
    MigratableDeviceBuffer2(AnyMigratable* migratable, int n = 0):
        Base(migratable, n) {}

    void allocate_elements_d(T** ptr, int count)
    {
        Base::get_migratable()->malloc_migratable_device((void**) ptr, count * sizeof(T));
    }

    void copy_elements_d(T* dst, T* src, int count)
    {
        CUDA_CHECK(cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyDeviceToDevice));
    }

    void set_zero_d(T* ptr, int count)
    {
        CUDA_CHECK(cudaMemset(ptr, 0, count * sizeof(T)));
    }

    void free_elements_d(T* ptr)
    {
        Base::get_migratable()->free_migratable(ptr);
    }
};

template <typename T>
class MigratableHostBuffer2 : public MigratableBufferBase2<T, MigratableHostBuffer2<T> >
{
private:
    typedef MigratableBufferBase2<T, MigratableHostBuffer2<T> > Base;
public:
    MigratableHostBuffer2(AnyMigratable* migratable, int n = 0):
        Base(migratable, n) {}

    void allocate_elements_d(T** ptr, int count)
    {
        Base::get_migratable()->malloc_migratable_host((void**) ptr, count * sizeof(T));
    }

    void copy_elements_d(T* dst, T* src, int count)
    {
        memcpy(dst, src, count * sizeof(T));
    }

    void set_zero_d(T* ptr, int count)
    {
        memset(ptr, 0, count * sizeof(T));
    }

    void free_elements_d(T* ptr)
    {
        Base::get_migratable()->free_migratable(ptr);
    }
};

template <typename T>
class MigratablePinnedBuffer2 : public MigratableBufferBase2<T, MigratablePinnedBuffer2<T> >
{
private:
    typedef MigratableBufferBase2<T, MigratablePinnedBuffer2<T> > Base;
public:
    MigratablePinnedBuffer2(AnyMigratable* migratable, int n = 0):
        Base(migratable, n) {}

    void allocate_elements_d(T** ptr, int count)
    {
        Base::get_migratable()->malloc_migratable_pinned((void**) ptr, count * sizeof(T));
    }

    void copy_elements_d(T* dst, T* src, int count)
    {
        CUDA_CHECK(cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyHostToHost));
    }

    void set_zero_d(T* ptr, int count)
    {
        CUDA_CHECK(cudaMemset(ptr, 0, count * sizeof(T)));
    }

    void free_elements_d(T* ptr)
    {
        Base::get_migratable()->free_migratable(ptr);
    }

    T* devptr()
    {
        if (MigratableBufferBase<T, MigratablePinnedBuffer2<T> >::data == NULL)
            return NULL;
        T* ptr;
        CUDA_CHECK(cudaHostGetDevicePointer(&ptr,
            MigratableBufferBase<T, MigratablePinnedBuffer2<T> >::data, 0));
        return ptr;
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
