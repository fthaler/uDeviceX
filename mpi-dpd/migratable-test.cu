#include <cassert>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <mpi.h>

#include "migration.h"

class MigratableThing : public Migratable
{
public:
    explicit MigratableThing() : n_ptrs(0), buffer_size(512), id(id) {}

    void alloc_host_data()
    {
        if (n_ptrs == MAX_BUFFERS)
            return;
        malloc_migratable_host(&ptrs[n_ptrs], buffer_size);
        on_device[n_ptrs++] = false;
    }

    void alloc_device_data()
    {
        if (n_ptrs == MAX_BUFFERS)
            return;
        malloc_migratable_device(&ptrs[n_ptrs], buffer_size);
        on_device[n_ptrs++] = true;
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

    void set_buffer_size(int new_buffer_size) { buffer_size = new_buffer_size; }
    void set_id(int new_id) { id = new_id; }
    int get_id() const { return id; }
private:
    void* ptrs[MAX_BUFFERS];
    bool on_device[MAX_BUFFERS];
    int n_ptrs, buffer_size, id;
};

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    srand(2342347 * rank);

    const int n = 7;
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
        switch (rand() % 3) {
        case 0:
            mt.alloc_host_data();
            std::cout << ".alloc_host_data()";
            break;
        case 1:
            mt.alloc_device_data();
            std::cout << ".alloc_device_data()";
            break;
        case 2:
            mt.free_last();
            std::cout << ".free_last()";
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
