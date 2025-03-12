#pragma once

#include <cstdlib>
#include <iterator>
#include <cstdint>

template<typename It>
struct BlockCyclicIt
{
    using iterator_category = std::random_access_iterator_tag;
    using difference_type   = std::ptrdiff_t;
    using value_type        = typename std::iterator_traits<It>::value_type;
    using reference         = value_type&;

    __host__ __device__
    BlockCyclicIt(It x_in, intptr_t s_in, intptr_t p_in, intptr_t b_in,
                  intptr_t k_in, intptr_t j_in) 
    {
        x = x_in;
        s = s_in;
        p = p_in;
        b = b_in;
        k = k_in;
        j = j_in;
    }

    __host__ __device__
    BlockCyclicIt(It x_in, intptr_t s_in, intptr_t p_in, intptr_t b_in)
    {
        x = x_in;
        s = s_in;
        p = p_in;
        b = b_in;
        k = 0;
        j = 0;
    }

    __host__ __device__
    reference operator*() const
    { 
        return *(x + (k * p + s) * b + j);
    }

    __host__ __device__
    BlockCyclicIt& operator+=(intptr_t count)
    {
        assert(count >= 0);
        /**
         * Let x be (k1p + s1)b + j1
         * Let y be (k2p + s2)b + j2
         *
         * Then 
         *
         * (k2 - k1) * b + j2 - j1 = count
         *
         * j2 = count + j1 - (k2 - k1) * b
         *
         * j2 mod b = (count + j1) mod b
         * j2       = (count + j1) mod b
         * 
         * (count + j1 >= 0, so we can use % for modulo)
         *
         * (k2 - k1) * b = count - (j2 - j1)
         *
         * k2 = k1 + (count - (j2 - j1)) / b
         **/
        intptr_t j1 = j;
        intptr_t k1 = k;
    
        intptr_t j2 = (count + j1) % b;
        assert((count - (j2 - j1)) % b == 0);
        intptr_t k2 = k1 + (count - (j2 - j1)) / b;
        
        j = j2;
        k = k2;
        assert(k2 >= 0);

        return *this;
    }

    __host__ __device__
    BlockCyclicIt& operator-=(intptr_t count)
    {
        assert(count >= 0);
        /**
         * Let x be (k1p + s1)b + j1
         * Let y be (k2p + s2)b + j2
         *
         * Then 
         *
         * (k1 - k2) * b + j1 - j2 = count
         *
         * j2 = j1 - count + (k1 - k2) * b
         *
         * j2 mod b = (j1 - count) mod b
         * j2       = (j1 - count) mod b
         * 
         * (k1 - k2) * b = count - (j1 - j2)
         *
         * k1 - k2 = (count - (j1 - j2)) / b
         * k2      = k1 - (count - (j1 - j2)) / b
         **/
        long j1 = j;
        long k1 = k;
    
        long j2 = ((j1 - count) % b + b) % b;
        long k2 = k1 - (count - (j1 - j2)) / b;
        
        j = j2;
        k = k2;
        assert(k2 >= 0);

        return *this;
    }

    __host__ __device__
    BlockCyclicIt& operator++()
    {
        (*this) += 1;
        return *this;
    }

    __host__ __device__
    BlockCyclicIt operator++(int) 
    {
        BlockCyclicIt tmp = *this; 
        (*this) += 1;
        return tmp;
    }

    __host__ __device__
    BlockCyclicIt& operator--()
    {
        (*this) -= 1;
        return *this;
    }

    __host__ __device__
    BlockCyclicIt operator+(intptr_t count)
    {
        intptr_t j1 = j;
        intptr_t k1 = k;
    
        intptr_t j2 = (count + j1) % b;
        intptr_t k2 = k1 + (count - (j2 - j1)) / b;

        return BlockCyclicIt(x, s, p, b, k2, j2);
    }

    __host__ __device__
    friend difference_type operator-(const BlockCyclicIt& a, 
                                     const BlockCyclicIt& b)
    {
        assert(a.s == b.s);
        return (a.k - b.k) * a.b + a.j - b.j;
    }

    __host__ __device__
    friend bool operator<(const BlockCyclicIt& a, const BlockCyclicIt& b)
    { 
        return (a.k * a.p + a.s) * a.b + a.j < (b.k * b.p + b.s) * b.b + b.j;
    }

    __host__ __device__
    friend bool operator==(const BlockCyclicIt& a, const BlockCyclicIt& b)
    { 
        bool same_arr = (a.s == b.s && a.b == b.b && a.p == b.p);
        bool same_ind = (a.k == b.k && a.j == b.j);
        return same_arr && same_ind;
    }

    __host__ __device__
    intptr_t get_global_index(void)
    { 
        return (k * p + s) * b + j;    
    }

    __host__ __device__
    intptr_t get_s(void)
    { 
        return s;
    }

    __host__ __device__
    intptr_t get_b(void)
    { 
        return b;
    }

    __host__ __device__
    intptr_t get_p(void)
    { 
        return p;
    }

    __host__ __device__
    intptr_t get_k(void)
    { 
        return k;
    }

    __host__ __device__
    intptr_t get_j(void)
    { 
        return j;
    }

    __device__
    void print(const char *prefix)
    { 
        printf("%s = %ld = (%ld * %ld + %ld) * %ld + %ld\n",
                prefix,
                (k * p + s) * b + j,
                k, p, s, b, j);

    }

public:
    It       x;
    intptr_t s; /* Processor index */
    intptr_t p; /* Number of processors */
    intptr_t b; /* Block size */
    intptr_t k; /* Block index */
    intptr_t j; /* Local index within block */
};

template<typename it>
__host__ __device__
inline BlockCyclicIt<it>
BlockCycSup(it x, intptr_t s, intptr_t p, intptr_t block, intptr_t n)
{
    /**
     * Compute n = (k * p + s) * block + j = n for 0 <= j < p * block.
     * if j < block, n is in I^(s) and we are done. If not, we need
     * to increase k by one.
     **/
    intptr_t k, j;
    if (n < s * block) {
        k = 0;
        j = 0;
    } else {
        k = (n - s * block) / (p * block);
        j = (n - s * block) % (p * block);
        if (j >= block) {
            j = 0;
            k += 1;
        }
    }
    
    assert(k >= 0);
    assert(j >= 0);
    assert((k * p + s) * block + j >= n);

    return BlockCyclicIt(x, s, p, block, k, j);
}
