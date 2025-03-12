#include <cuda.h>
#include <cub/cub.cuh>
#include <assert.h>

#include "util.h"
#include "blockcyclic.h"

using namespace cub;

template<typename T>
struct scratch_space {
    /* Buffers so we can do the rough partition in-place. */
    T *buf_left;
    T *buf_right;
    /* Local splits for each block-cyclic subarray */
    intptr_t *splits;
    /* Number of incorrect elements for each block-cyclic subarray.
     * Positive if we have too few elements satisfying pred, negative
     * if we have too many. */
    intptr_t *to_move;
    unsigned long long *to_move_total;
    /* Partition to_move in < 0, >= 0. */
    char     *to_move_flags;
    intptr_t *to_move_index;
    intptr_t *to_move_split;
    intptr_t *L_proc;
    intptr_t *L_off;
    intptr_t *R_proc;
    intptr_t *R_off;
    /* We cut off a small part so the largest part of the array
     * is divisible by TILE_SIZE. m2 is the split of this small part. */
    intptr_t *m2;
    /* Global split of the large part. */
    unsigned long long *m;
    void *cub_temp_storage;
    size_t cub_storage_size;
};

template<typename T, int BLOCKS, int TILE_SIZE>
size_t alloc_needed(void)
{
    void *d_temp_storage = nullptr;
    size_t storage_partition = 0;
    /* To silence warnings. This memory is not accessed, it is only
     * so the template can find the type. */
    intptr_t *to_move       = (intptr_t *)0xfa1afe1;
    char *to_move_flags     = (char *)0xfa1afe1;
    intptr_t *to_move_split = (intptr_t *)0xfa1afe1;
    DevicePartition::Flagged(d_temp_storage, storage_partition,
                             to_move, to_move_flags,
                             to_move, to_move_split,
                             BLOCKS);
    d_temp_storage = nullptr;
    size_t storage_prefix = 0;
    DeviceScan::InclusiveSum(d_temp_storage, storage_prefix,
                             to_move, to_move, BLOCKS);
    size_t cub_storage_size = max(storage_partition, storage_prefix);

    return cub_storage_size +
           BLOCKS * TILE_SIZE * sizeof(T) +
           BLOCKS * TILE_SIZE * sizeof(T) +
           BLOCKS           * sizeof(intptr_t) +
           BLOCKS           * sizeof(intptr_t) +
           1                * sizeof(unsigned long long) +
           BLOCKS           * sizeof(char) +
           BLOCKS           * sizeof(intptr_t) +
           1                * sizeof(intptr_t) +
           BLOCKS           * sizeof(intptr_t) +
           BLOCKS           * sizeof(intptr_t) +
           BLOCKS           * sizeof(intptr_t) +
           BLOCKS           * sizeof(intptr_t) +
           1                * sizeof(intptr_t) +
           1                * sizeof(unsigned long long);
}

template<typename T, int BLOCKS, int TILE_SIZE>
struct scratch_space<T> get_scratch(void *buf, size_t *d_m)
{
    struct scratch_space<T> scratch;

    void *d_temp_storage = nullptr;
    size_t storage_partition = 0;
    /* To silence warnings. This memory is not accessed, it is only
     * so the template can find the type. */
    intptr_t *to_move       = (intptr_t *)0xfa1afe1;
    char *to_move_flags     = (char *)0xfa1afe1;
    intptr_t *to_move_split = (intptr_t *)0xfa1afe1;
    DevicePartition::Flagged(d_temp_storage, storage_partition,
                             to_move, to_move_flags,
                             to_move, to_move_split,
                             BLOCKS);
    d_temp_storage = nullptr;
    size_t storage_prefix = 0;
    DeviceScan::InclusiveSum(d_temp_storage, storage_prefix,
                             to_move, to_move, BLOCKS);
    scratch.cub_storage_size = max(storage_partition, storage_prefix);

    scratch.buf_left  = (T *)buf;
    scratch.buf_right = (T *)(scratch.buf_left         + BLOCKS * TILE_SIZE);
    scratch.splits    = (intptr_t *)(scratch.buf_right + BLOCKS * TILE_SIZE);
    scratch.to_move   = (intptr_t *)(scratch.splits    + BLOCKS);
    scratch.to_move_total = (unsigned long long *)(scratch.to_move + BLOCKS);
    scratch.to_move_flags = (char     *)(scratch.to_move_total + 1);
    scratch.to_move_index = (intptr_t *)(scratch.to_move_flags + BLOCKS);
    scratch.to_move_split = (intptr_t *)(scratch.to_move_index + BLOCKS);
    scratch.L_proc        = (intptr_t *)(scratch.to_move_split + 1);
    scratch.L_off         = (intptr_t *)(scratch.L_proc + BLOCKS);
    scratch.R_proc        = (intptr_t *)(scratch.L_off  + BLOCKS);
    scratch.R_off         = (intptr_t *)(scratch.R_proc + BLOCKS);
    scratch.m2            = (intptr_t *)(scratch.R_off  + BLOCKS);
    scratch.cub_temp_storage = (void *)(scratch.m2 + 1);
    scratch.m             = (unsigned long long *)(d_m);

    return scratch;
}

/**
 * Cooperatively read in global memory to local.
 **/
template<typename It, typename T>
inline __device__ void
read_local(T *dest, BlockCyclicIt<It> src, int count)
{
    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        assert(src.j == 0);
        assert(count <= src.b);
        dest[i] = *(src.x + (src.k * src.p + src.s) * src.b + i);
    }
}

/**
 * Cooperatively read in global memory to registers.
 **/
template<typename It, typename T>
inline __device__ void
read_reg(T dest[], BlockCyclicIt<It> src, int items_per_thread)
{
    intptr_t offset = (src.k * src.p + src.s) * src.b + threadIdx.x;
    #pragma unroll
    for (int i = 0; i < items_per_thread; i++) {
        assert(src.j == 0);
        dest[i] = *(src.x + offset);
        offset += blockDim.x;
    }
}

template<typename Predicate, typename T, int ITEMS_PER_THREAD, 
         int THREADS_PER_BLOCK>
inline __device__ int
local_partition(T (&thread_x)[ITEMS_PER_THREAD], Predicate pred)
{
    int index[ITEMS_PER_THREAD];

    using BlockScan = BlockScan<int, THREADS_PER_BLOCK>;
    static __shared__ typename BlockScan::TempStorage temp_scan;
    using BlockExchange = BlockExchange<T, THREADS_PER_BLOCK,
                                        ITEMS_PER_THREAD>;
    static __shared__ typename BlockExchange::TempStorage temp_ex;

    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        index[i] = pred(thread_x[i]);
    }
    int total_left;
    BlockScan(temp_scan).ExclusiveSum(index, index, total_left);
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        if (!pred(thread_x[i])) {
            index[i] = ITEMS_PER_THREAD * THREADS_PER_BLOCK - 1 -
                       (threadIdx.x * ITEMS_PER_THREAD + i - index[i]);
        }
    }
    BlockExchange(temp_ex).ScatterToStriped(thread_x, index);

#ifndef NDEBUG
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int j = i * blockDim.x + threadIdx.x;
        assert(pred(thread_x[i]) || j >= total_left);
        assert(!pred(thread_x[i]) || j < total_left);
    }
#endif

    return total_left;
}

template<typename Predicate, typename It, typename T, int ITEMS_PER_THREAD>
inline __device__ void
local_write(BlockCyclicIt<It> &writeL, BlockCyclicIt<It> &writeR,
            T (&thread_x)[ITEMS_PER_THREAD],
            int total_left, int total_right, intptr_t n, Predicate pred)
{
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int j = i * blockDim.x + threadIdx.x;
        if (j < total_left) {
            assert(pred(thread_x[i]));
            assert((writeL + j).get_global_index() < n);
            *(writeL + j) = thread_x[i];
        }
    }
    writeL += total_left;
    writeR -= total_right;
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int j = i * blockDim.x + threadIdx.x;
        if (j >= total_left) {
            assert(!pred(thread_x[i]));
            assert((writeR + (j - total_left)).get_global_index() < n);
            assert((writeR + (j - total_left)).get_global_index() >= 0);
            *(writeR + (j - total_left)) = thread_x[i];
            assert(!pred(*(writeR + (j - total_left))));
        }
    }
}

template<typename Predicate, typename It, typename T, int ITEMS_PER_THREAD, 
         int THREADS_PER_BLOCK, int TILE_SIZE>
inline __device__ void
loop_body(BlockCyclicIt<It> &writeL, BlockCyclicIt<It> &writeR,
          T (&thread_x)[ITEMS_PER_THREAD], intptr_t n, Predicate pred)
{
    int total_left  = local_partition<Predicate, T, ITEMS_PER_THREAD, 
                                      THREADS_PER_BLOCK>
                                      (thread_x, pred);
    int total_right = TILE_SIZE - total_left;

    local_write<Predicate, It, T, ITEMS_PER_THREAD>
               (writeL, writeR, thread_x, total_left, total_right, n, pred);
    __syncthreads(); /* Necessary for reusing cub buffers */
}

template<typename Predicate, typename It, typename T, int BLOCKS, 
         int ITEMS_PER_THREAD, int THREADS_PER_BLOCK, int TILE_SIZE>
__global__ void
partition_rough(intptr_t n, It x, Predicate pred,
                struct scratch_space<T> scratch)
{
    /**
     * We are not going to handle the case n % TILE_SIZE != 0 in this kernel
     * call because it increases the pressure on registers and local memory.
     **/
    assert(n % TILE_SIZE == 0);

    using value_type = typename std::iterator_traits<It>::value_type;

    value_type thread_x[ITEMS_PER_THREAD];

    value_type *buf_left  = scratch.buf_left;
    value_type *buf_right = scratch.buf_right;
    intptr_t *splits      = scratch.splits;

    BlockCyclicIt readL(x, blockIdx.x, BLOCKS, TILE_SIZE);
    BlockCyclicIt readR  = BlockCycSup(x, blockIdx.x, BLOCKS, TILE_SIZE, n);
    BlockCyclicIt writeL = readL;
    BlockCyclicIt writeR = readR;

    intptr_t my_n = readR - readL;

    assert(my_n >= 0);
    assert(my_n % TILE_SIZE == 0);

    if (my_n == 0) {
        goto done;
    }

    /**
     * Buffer a block / tile on the left and right so that we can partition
     * in-place.
     **/
    assert((readL + (TILE_SIZE - 1)).get_global_index() < n);
    read_local(buf_left + blockIdx.x * TILE_SIZE, readL, TILE_SIZE);
    readL.k++; // readL += TILE_SIZE
    if (my_n == TILE_SIZE) {
        goto write_left;
    }
    readR.k--; // readR -= TILE_SIZE;
    read_local(buf_right + blockIdx.x * TILE_SIZE, readR, TILE_SIZE);

    /**
     * Block cyclic partition of block-cyclic subarray.
     **/
    while (readR - readL >= TILE_SIZE) {
        intptr_t cap_left  = readL - writeL;
        intptr_t cap_right = writeR - readR;
        if (cap_left <= cap_right) {
            read_reg(thread_x, readL, ITEMS_PER_THREAD);
            readL.k++; // readL += TILE_SIZE;
        } else {
            readR.k--; // readR -= TILE_SIZE;
            read_reg(thread_x, readR, ITEMS_PER_THREAD);
        }

        loop_body<Predicate, It, value_type, ITEMS_PER_THREAD, 
                  THREADS_PER_BLOCK, TILE_SIZE>
                 (writeL, writeR, thread_x, n, pred);
    }

    /**
     * Partition buffers
     **/
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        thread_x[i] = buf_right[blockIdx.x * TILE_SIZE +
                                i * blockDim.x + threadIdx.x];
    }
    loop_body<Predicate, It, value_type, ITEMS_PER_THREAD, 
              THREADS_PER_BLOCK, TILE_SIZE>
             (writeL, writeR, thread_x, n, pred);

write_left:
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        thread_x[i] = buf_left[blockIdx.x * TILE_SIZE +
                               i * blockDim.x + threadIdx.x];
    }
    loop_body<Predicate, It, value_type, ITEMS_PER_THREAD, 
              THREADS_PER_BLOCK, TILE_SIZE>
             (writeL, writeR, thread_x, n, pred);

done:
    if (threadIdx.x == 0) {
        splits[blockIdx.x] = writeL.get_global_index();
        *(scratch.m) = 0;
        *(scratch.to_move_total) = 0;
    }
}

/**
 * Partitions x[0, n) for n < ITEMS_PER_THREAD * THREADS_PER_BLOCK
 **/
template<typename Predicate, typename It, int ITEMS_PER_THREAD,
         int THREADS_PER_BLOCK>
__device__ int
partition_partial(intptr_t n, It x, Predicate pred)
{
    using value_type = typename std::iterator_traits<It>::value_type;

    int        index[ITEMS_PER_THREAD];
    value_type thread_x[ITEMS_PER_THREAD];

    using BlockLoad = BlockLoad<value_type, THREADS_PER_BLOCK,
                                ITEMS_PER_THREAD, BLOCK_LOAD_WARP_TRANSPOSE>;
    __shared__ typename BlockLoad::TempStorage temp_load;
    using BlockScan = BlockScan<int, THREADS_PER_BLOCK>;
    static __shared__ typename BlockScan::TempStorage temp_scan;
    using BlockExchange = BlockExchange<value_type, THREADS_PER_BLOCK,
                                        ITEMS_PER_THREAD>;
    __shared__ typename BlockExchange::TempStorage temp_ex;

    BlockLoad(temp_load).Load(x, thread_x, n);

    /* Implicit transpose because prefix sum wants blocked */
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int j = threadIdx.x * ITEMS_PER_THREAD + i;
        index[i] = pred(thread_x[i]);
        if (j >= n) {
            index[i] = 0;
        }
    }

    int total_left;
    BlockScan(temp_scan).ExclusiveSum(index, index, total_left);
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int j = threadIdx.x * ITEMS_PER_THREAD + i;
        if (!pred(thread_x[i])) {
            index[i] = n - 1 - (j - index[i]);
        }
        if (j >= n) {
            index[i] = -1;
        }
    }
    BlockExchange(temp_ex).ScatterToStripedGuarded(thread_x, index);

#ifndef NDEBUG
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int j = i * blockDim.x + threadIdx.x;
        if (j < n) {
            assert(pred(thread_x[i]) || j >= total_left);
            assert(!pred(thread_x[i]) || j < total_left);
        }
    }
#endif

    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int j = i * blockDim.x + threadIdx.x;
        if (j < n) {
            x[j] = thread_x[i];
        }
    }

    return total_left;
}

/**
 * Cleanup superstep 1: partition the n_leftover < TILE_SIZE elements on
 * the right hand-side and compute m.
 **/
template<typename Predicate, typename It, typename T, int BLOCKS, 
         int ITEMS_PER_THREAD, int THREADS_PER_BLOCK, int TILE_SIZE>
__global__ void
cleanup_1(intptr_t n, It x, intptr_t n_leftover, Predicate pred,
          struct scratch_space<T> scratch)
{
    using BlockReduce = BlockReduce<unsigned long long, THREADS_PER_BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp_sum;

    intptr_t *splits = scratch.splits;

    if (blockIdx.x == gridDim.x - 1) {
        intptr_t m2 = partition_partial<Predicate, It, ITEMS_PER_THREAD,
                                        THREADS_PER_BLOCK>
                                       (n_leftover, x + n, pred);
        if (threadIdx.x == 0) {
            *(scratch.m2) = m2;
        }
    } else {
        unsigned long long m = 0;
        for (int s = blockIdx.x * blockDim.x + threadIdx.x;
             s < BLOCKS;
             s += (gridDim.x - 1) * blockDim.x)
        {
            BlockCyclicIt start(x, s, BLOCKS, TILE_SIZE);
            BlockCyclicIt end = BlockCycSup(x, s, BLOCKS, TILE_SIZE, splits[s]);
            m += (unsigned long long)(end - start);
        }
        m = BlockReduce(temp_sum).Sum(m);

        if (threadIdx.x == 0) {
            (void)atomicAdd(scratch.m, m);
        }
    }
}

/**
 * Compute to_move, set flags for < 0, >= 0, sum_s |to_move[s]|,
 * prepare permutation vector by setting it to [0, ..., BLOCKS - 1].
 **/
template<typename It, typename T, int BLOCKS, int THREADS_PER_BLOCK, 
         int TILE_SIZE>
__global__ void
cleanup_2(It x, struct scratch_space<T> scratch)
{
    using BlockReduce = BlockReduce<unsigned long long, THREADS_PER_BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp_sum;

    intptr_t *to_move       = scratch.to_move;
    char     *to_move_flags = scratch.to_move_flags;
    intptr_t *splits        = scratch.splits;
    intptr_t *to_move_index = scratch.to_move_index;
    intptr_t  m             = (intptr_t)*(scratch.m);

    unsigned long long to_move_total = 0;
    for (int s = blockIdx.x * blockDim.x + threadIdx.x;
         s < BLOCKS;
         s += gridDim.x * blockDim.x)
    {
        BlockCyclicIt m_ind  = BlockCycSup(x, s, BLOCKS, TILE_SIZE, m);
        BlockCyclicIt ms_ind = BlockCycSup(x, s, BLOCKS, TILE_SIZE, splits[s]);
        assert(m_ind.get_global_index() >= m);
        BlockCyclicIt begin(x, s, BLOCKS, TILE_SIZE);
        intptr_t left_before = ms_ind - begin;
        intptr_t left_after  = m_ind  - begin;
        to_move_total   += (unsigned long long)abs(left_after - left_before);
        to_move[s]       =  left_after - left_before;
        to_move_flags[s] = (left_after - left_before < 0);
        scratch.to_move_index[s] = s;
        assert(to_move[s] >= 0 || splits[s] > m);
    }

    to_move_total = BlockReduce(temp_sum).Sum(to_move_total);
    if (threadIdx.x == 0) {
        atomicAdd(scratch.to_move_total, to_move_total);
    }
}

/**
 * Cleanup superstep 3: partition to_move and splits
 **/
template<typename T, int BLOCKS>
void cleanup_3(struct scratch_space<T> scratch)
{
    DevicePartition::Flagged(scratch.cub_temp_storage,
                             scratch.cub_storage_size,
                             scratch.to_move, scratch.to_move_flags,
                             scratch.to_move, scratch.to_move_split,
                             BLOCKS);
    /* TODO: Pretty sure cub has already computed this internally,
     * can we extract this somehow? */
    DevicePartition::Flagged(scratch.cub_temp_storage,
                             scratch.cub_storage_size,
                             scratch.to_move_index, scratch.to_move_flags,
                             scratch.to_move_index, scratch.to_move_split,
                             BLOCKS);
    DevicePartition::Flagged(scratch.cub_temp_storage,
                             scratch.cub_storage_size,
                             scratch.splits, scratch.to_move_flags,
                             scratch.splits, scratch.to_move_split,
                             BLOCKS);
}

/**
 * Compute prefix sum of the positive and negative parts of to_move
 **/
template<typename T, int BLOCKS>
void cleanup_4(struct scratch_space<T> scratch)
{
    intptr_t to_move_split;
    CUDA_SAFE(cudaMemcpy(&to_move_split, scratch.to_move_split,
                         sizeof(intptr_t), cudaMemcpyDeviceToHost));

    DeviceScan::InclusiveSum(scratch.cub_temp_storage,
                             scratch.cub_storage_size,
                             scratch.to_move, scratch.to_move, to_move_split);
    DeviceScan::InclusiveSum(scratch.cub_temp_storage,
                             scratch.cub_storage_size,
                             scratch.to_move + to_move_split,
                             scratch.to_move + to_move_split,
                             BLOCKS - to_move_split);
}

/**
 * Computes start points of the chunks that still have to be swapped.
 **/
template<typename T, int BLOCKS>
__global__ void
cleanup_5(struct scratch_space<T> scratch)
{
    intptr_t split = *(scratch.to_move_split);
    intptr_t t = blockIdx.x * blockDim.x + threadIdx.x;
    intptr_t C = CeilDiv(*(scratch.to_move_total) / 2, BLOCKS);

    if (t >= BLOCKS) return;

    intptr_t prefix_low  = (t == 0 || t == split) ?
                                0 :
                                abs(scratch.to_move[t - 1]);
    intptr_t prefix_high = abs(scratch.to_move[t]);

    assert(t != BLOCKS - 1 || prefix_high == *(scratch.to_move_total) / 2);
    assert(t != split  - 1 || prefix_high == *(scratch.to_move_total) / 2);

    for (intptr_t s = CeilDiv(prefix_low, C);
                  s < CeilDiv(prefix_high, C); s++)
    {
        if (t < split) {
            scratch.L_proc[s] = t;
            scratch.L_off[s]  = s * C - prefix_low;
        } else {
            scratch.R_proc[s] = t;
            scratch.R_off[s]  = s * C - prefix_low;
        }
    }
}

/**
 * Swap [i, i + count) with [j, j + count). Asserts [i, i + count)
 * satisfies !pred and [j, j + count) satisfies pred.
 **/
template<typename Predicate, typename It>
inline __device__
void swap(BlockCyclicIt<It> &i, BlockCyclicIt<It> &j, intptr_t count,
          Predicate pred)
{
    assert(count >= 0);
    using value_type = typename std::iterator_traits<It>::value_type;

    intptr_t swapped_elems = count % blockDim.x;
    if (threadIdx.x < swapped_elems) {
        assert(!pred(*(i + threadIdx.x)));
        assert( pred(*(j + threadIdx.x)));
        value_type swap = *(i + threadIdx.x);
        *(i + threadIdx.x) = *(j + threadIdx.x);
        *(j + threadIdx.x) = swap;
    }
    i += swapped_elems;
    j += swapped_elems;

    for (; swapped_elems < count; swapped_elems += blockDim.x) {
        assert(!pred(*(i + threadIdx.x)));
        assert( pred(*(j + threadIdx.x)));
        value_type swap = *(i + threadIdx.x);
        *(i + threadIdx.x) = *(j + threadIdx.x);
        *(j + threadIdx.x) = swap;
        i += blockDim.x;
        j += blockDim.x;
    }
}

/**
 * Cleanup superstep 6, does the actual swapping
 **/
template<typename Predicate, typename It, typename T, int BLOCKS, int TILE_SIZE>
__global__ void
cleanup_6(It x, Predicate pred, struct scratch_space<T> scratch)
{
    intptr_t to_move_total = *(scratch.to_move_total) / 2;
    intptr_t C = CeilDiv(to_move_total, BLOCKS);

    if (blockIdx.x * C >= to_move_total) return;

    intptr_t *splits = scratch.splits;
    intptr_t *to_move_index = scratch.to_move_index;

    intptr_t m = (intptr_t)*(scratch.m);
    intptr_t bleft  = scratch.L_proc[blockIdx.x];
    intptr_t bright = scratch.R_proc[blockIdx.x];
    BlockCyclicIt left  = BlockCycSup<It>(x, to_move_index[bleft],
                                      (intptr_t)BLOCKS, TILE_SIZE, m);
    BlockCyclicIt right = BlockCycSup(x, to_move_index[bright],
                                      (intptr_t)BLOCKS, TILE_SIZE,
                                      splits[bright]);
    left  += scratch.L_off[blockIdx.x];
    right += scratch.R_off[blockIdx.x];

    intptr_t start         = MIN( blockIdx.x      * C, to_move_total);
    intptr_t end           = MIN((blockIdx.x + 1) * C, to_move_total);
    intptr_t to_swap       = end - start;
    intptr_t total_swapped = 0;
    while (total_swapped < to_swap) {
        BlockCyclicIt endL = BlockCycSup(x, to_move_index[bleft],
                                         (intptr_t)BLOCKS,
                                         TILE_SIZE,
                                         splits[bleft]);
        BlockCyclicIt endR = BlockCycSup(x, to_move_index[bright],
                                         (intptr_t)BLOCKS, TILE_SIZE, m);
        intptr_t count = min(endL - left, endR - right);
        count = min(count, to_swap - total_swapped);
        swap(right, left, count, pred);
        total_swapped += count;
        if (total_swapped == to_swap) break;
        if (!(left < endL)) {
            bleft++;
            left = BlockCycSup(x, to_move_index[bleft], (intptr_t)BLOCKS,
                               TILE_SIZE, m);
        }
        if (!(right < endR)) {
            bright++;
            right = BlockCycSup(x, to_move_index[bright], (intptr_t)BLOCKS,
                                TILE_SIZE,
                                splits[bright]);
        }
    }
    assert(total_swapped == to_swap);
}

/**
 * Cleanup superstep 7, swap the final n_leftover < TILE_SIZE into globally
 * correct position.
 *
 *  Writing L(eft) for elements satisfying pred and R(ight) for those
 *  that don't, we have this.
 *
 * |     L      |    R    |   L    |    R   |
 *             /\        /\        /\      /\
 *             m          n      n + m2   n + n_leftover
 *
 *
 *  We swap ranges so we get
 *
 * | L | R |
 *
 * and update m.
 **/
template <typename It, typename T>
__global__ void
cleanup_7(intptr_t n, It x, intptr_t n_leftover,
           struct scratch_space<T> scratch)
{
    using value_type = typename std::iterator_traits<It>::value_type;
    intptr_t m =  *(scratch.m);
    intptr_t m2 = *(scratch.m2);
    intptr_t swap_sz = min(n - m, m2);

    for (int i = threadIdx.x; i < swap_sz; i += blockDim.x) {
        value_type temp = *(x + (m + i));
        *(x + (m + i)) = *(x + (n + m2 - swap_sz + i));
        *(x + (n + m2 - swap_sz + i)) = temp;
    }

    *(scratch.m) = m + m2;
}

template<typename Predicate, typename It, int BLOCKS = 4096, 
         int ITEMS_PER_THREAD = 4, int THREADS_PER_BLOCK = 128>
void partition(void *d_temp_storage, size_t &temp_storage_bytes,
               It begin, It end, size_t *d_num_selected_out, Predicate pred)
{
    using value_type = typename std::iterator_traits<It>::value_type;
    constexpr int TILE_SIZE = (intptr_t)ITEMS_PER_THREAD * THREADS_PER_BLOCK;

    if (d_temp_storage == NULL) {
        temp_storage_bytes = alloc_needed<value_type, BLOCKS, TILE_SIZE>();
        return;
    }

    intptr_t n = end - begin;
    intptr_t n_leftover = n % TILE_SIZE;
    n = n - n_leftover;

    struct scratch_space<value_type> scratch = 
            get_scratch<value_type, BLOCKS, TILE_SIZE>
                       (d_temp_storage, d_num_selected_out);

    partition_rough<Predicate, It, value_type, BLOCKS, ITEMS_PER_THREAD,
                    THREADS_PER_BLOCK, TILE_SIZE>
                   <<<BLOCKS, THREADS_PER_BLOCK>>>
                   (n, begin, pred, scratch);
    int cl_blocks = CeilDiv(BLOCKS, THREADS_PER_BLOCK);
    cleanup_1<Predicate, It, value_type, BLOCKS, ITEMS_PER_THREAD, 
              THREADS_PER_BLOCK, TILE_SIZE>
             <<<cl_blocks + 1, THREADS_PER_BLOCK>>>
             (n, begin, n_leftover, pred, scratch);
    cleanup_2<It, value_type, BLOCKS, THREADS_PER_BLOCK, TILE_SIZE>
             <<<cl_blocks, THREADS_PER_BLOCK>>>(begin, scratch);
    cleanup_3<value_type, BLOCKS>(scratch);
    cleanup_4<value_type, BLOCKS>(scratch);
    cleanup_5<value_type, BLOCKS>
             <<<cl_blocks, THREADS_PER_BLOCK>>>
             (scratch);
    cleanup_6<Predicate, It, value_type, BLOCKS, TILE_SIZE>
             <<<BLOCKS, THREADS_PER_BLOCK>>>
             (begin, pred, scratch);
    cleanup_7<<<1, THREADS_PER_BLOCK>>>
             (n, begin, n_leftover, scratch);
    return;
}
