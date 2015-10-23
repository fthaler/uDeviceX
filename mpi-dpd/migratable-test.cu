#include <cassert>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <mpi.h>

#include "migration.h"
#include "migratable-datastructures.h"

class MigratableThing : public Migratable<20>
{
public:
    explicit MigratableThing() : n_ptrs(0), buffer_size(512), id(id),
        dev_buffer2(this), host_buffer2(this), pin_buffer2(this) {}

    void alloc_some_data(int type)
    {
        assert(type >= 0 && type < 4);
        if (n_ptrs >= MAX_BUFFERS - 4)
            return;
        switch (type) {
        case 0:
            malloc_migratable(&ptrs[n_ptrs], buffer_size);
            on_device[n_ptrs++] = false;
            break;
        case 1:
            malloc_migratable_device(&ptrs[n_ptrs], buffer_size);
            on_device[n_ptrs++] = true;
            break;
        case 2:
            malloc_migratable_host(&ptrs[n_ptrs], buffer_size);
            on_device[n_ptrs++] = false;
            break;
        case 3:
            malloc_migratable_pinned(&ptrs[n_ptrs], buffer_size);
            on_device[n_ptrs++] = false;
            break;
        }
    }

    void do_some_stuff()
    {
        for (int i = 0; i < n_ptrs; ++i) {
            if (on_device[i])
                cudaMemset(ptrs[i], 0, buffer_size);
            else
                memset(ptrs[i], 0, buffer_size);
        }
    }

    void free_last()
    {
        if (n_ptrs == 0)
            return;
        free_migratable(ptrs[--n_ptrs]);
    }

    void resize_device_buffer(int size)
    {
        dev_buffer.resize(size);
    }

    void preserve_resize_device_buffer(int size)
    {
        dev_buffer.preserve_resize(size);
    }

    void resize_host_buffer(int size)
    {
        host_buffer.resize(size);
    }

    void preserve_resize_host_buffer(int size)
    {
        host_buffer.preserve_resize(size);
    }

    void resize_pinned_buffer(int size)
    {
        pin_buffer.resize(size);
    }

    void preserve_resize_pinned_buffer(int size)
    {
        pin_buffer.preserve_resize(size);
    }

    void resize_device_buffer2(int size)
    {
        dev_buffer2.resize(size);
    }

    void preserve_resize_device_buffer2(int size)
    {
        dev_buffer2.preserve_resize(size);
    }

    void resize_host_buffer2(int size)
    {
        host_buffer2.resize(size);
    }

    void preserve_resize_host_buffer2(int size)
    {
        host_buffer2.preserve_resize(size);
    }

    void resize_pinned_buffer2(int size)
    {
        pin_buffer2.resize(size);
    }

    void preserve_resize_pinned_buffer2(int size)
    {
        pin_buffer2.preserve_resize(size);
    }

    void set_buffer_size(int new_buffer_size) { buffer_size = new_buffer_size; }
    void set_id(int new_id) { id = new_id; }
    int get_id() const { return id; }
private:
    void* ptrs[MAX_BUFFERS];
    bool on_device[MAX_BUFFERS];
    int n_ptrs, buffer_size, id;

    MigratableDeviceBuffer<int> dev_buffer;
    MigratableHostBuffer<float> host_buffer;
    MigratablePinnedBuffer<double> pin_buffer;

    MigratableDeviceBuffer2<int> dev_buffer2;
    MigratableHostBuffer2<float> host_buffer2;
    MigratablePinnedBuffer2<double> pin_buffer2;
};

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    srand(2342347 * rank);

    const int n = 1;
    MigratableThing things[n];
    for (int i = 0; i < n; ++i) {
        things[i].set_buffer_size(rand() % 1024);
        things[i].set_id(i + rank * n);
    }

    for (int i = 0; i < 100; ++i)  {
        char processor_name[MPI_MAX_PROCESSOR_NAME];
        int processor_name_length;
        MPI_Get_processor_name(processor_name, &processor_name_length);
        processor_name[processor_name_length + 1] = 0;
        std::cout << processor_name << ": ";

        int j = rand() % n;
        MigratableThing& mt = things[j];
        std::cout << "things[" << mt.get_id() << "]";
        int t = rand() % 14;
        int size = 0;
        switch (t) {
        case 0:
        case 1:
        case 2:
        case 3:
            mt.alloc_some_data(t);
            std::cout << ".alloc_some_data(" << t << ")";
            break;
        case 4:
            mt.free_last();
            std::cout << ".free_last()";
            break;
        case 5:
            size = rand() % 1024;
            mt.resize_device_buffer(size);
            std::cout << ".resize_device_buffer(" << size << ")";
            break;
        case 6:
            size = rand() % 1024;
            mt.preserve_resize_device_buffer(size);
            std::cout << ".preserve_resize_device_buffer(" << size << ")";
            break;
        case 7:
            size = rand() % 1024;
            mt.resize_host_buffer(size);
            std::cout << ".resize_host_buffer(" << size << ")";
            break;
        case 8:
            size = rand() % 1024;
            mt.preserve_resize_host_buffer(size);
            std::cout << ".preserve_resize_host_buffer(" << size << ")";
            break;
        case 9:
            size = rand() % 1024;
            mt.resize_pinned_buffer(size);
            std::cout << ".resize_pinned_buffer(" << size << ")";
            break;
        case 10:
            size = rand() % 1024;
            mt.preserve_resize_pinned_buffer(size);
            std::cout << ".preserve_resize_pinned_buffer(" << size << ")";
            break;
        case 11:
            size = rand() % 1024;
            mt.preserve_resize_host_buffer2(size);
            std::cout << ".preserve_resize_host_buffer2(" << size << ")";
            break;
        case 12:
            size = rand() % 1024;
            mt.resize_pinned_buffer2(size);
            std::cout << ".resize_pinned_buffer2(" << size << ")";
            break;
        case 13:
            size = rand() % 1024;
            mt.preserve_resize_pinned_buffer2(size);
            std::cout << ".preserve_resize_pinned_buffer2(" << size << ")";
            break;
        }
        std::cout << std::endl;

        for (int j = 0; j < n; ++j)
            things[j].do_some_stuff();

#ifdef AMPI
        MPI_Migrate();
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0)
            std::cout << "--- MIGRATION ---" << std::endl;
        MPI_Barrier(MPI_COMM_WORLD);
#endif
    }
    

    MPI_Finalize();
    return 0;
};
