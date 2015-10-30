#pragma once

#include <algorithm>
#include <cassert>
#include <functional>
#include <cstring>

template <typename T, unsigned S>
class StackAllocator
{
public:
    typedef T value_type;
    typedef T* pointer;
    typedef const T* const_pointer;
    typedef T& reference;
    typedef const T& const_reference;
    typedef unsigned size_type;
    typedef int difference_type;

    template <typename U>
    struct rebind {
        typedef StackAllocator<U, S> other;
    };

    StackAllocator() { std::memset(data, 0, CHAR_COUNT); }
    template <typename U, size_type R>
    StackAllocator(const StackAllocator<U, R>& other) { std::memset(data, 0, CHAR_COUNT); }

    pointer allocate(size_type n)
    {
        const unsigned alloc_size = allocated_size(n);

        int i = 0;
        int last_free = 0;
        while (i < SIZE_TYPE_COUNT) {
            if (data[i] == 0) {
                ++last_free;
                ++i;
                if (last_free == alloc_size) {
                    const int j = i - alloc_size;
                    data[j] = alloc_size;
                    return reinterpret_cast<pointer>(&data[j + 1]);
                }
            } else {
                i += data[i];
                last_free = 0;
            }
        }
        throw std::bad_alloc();
    }

    void deallocate(pointer ptr, size_type n)
    {
        size_type* data_ptr = reinterpret_cast<size_type*>(ptr) - 1;
        const unsigned alloc_size = allocated_size(n);
        assert(data_ptr >= &data[0] && data_ptr < &data[SIZE_TYPE_COUNT]);
        assert(*data_ptr == alloc_size);
        std::memset(data_ptr, 0, alloc_size * sizeof(size_type));
    }

    void construct(pointer ptr, const_reference val)
    {
        new (reinterpret_cast<void*>(ptr)) value_type(val);
    }

    void destroy(pointer ptr)
    {
        ptr->~value_type();
    }

    size_type max_size() const
    {
        return VALUE_TYPE_COUNT;
    }

    template <typename U, size_type R>
    bool operator== (const StackAllocator<U, R>&) const { return false; }
    template <typename U, size_type R>
    bool operator!= (const StackAllocator<U, R>&) const { return true; }

private:
    size_type allocated_size(size_type n) const
    {
        const unsigned alloc_bytes = sizeof(value_type) * n + sizeof(size_type);
        return (alloc_bytes + sizeof(size_type) - 1) / sizeof(size_type);
    }

    enum { VALUE_TYPE_COUNT = S };
    enum { CHAR_COUNT = S * sizeof(value_type) };
    enum { SIZE_TYPE_COUNT = S * sizeof(value_type) / sizeof(size_type) };

    size_type data[SIZE_TYPE_COUNT];
};
