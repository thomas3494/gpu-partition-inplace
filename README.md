# Usage

This is a header-only library for partitioning in-place on a GPU, built on top
of [CUB](https://github.com/nvidia/cccl).
The API is similar to [cub::partition](https://nvidia.github.io/cccl/cub/api/structcub_1_1DevicePartition.html#_CPPv4N3cub15DevicePartitionE), except for
three additional template parameters 
`BLOCKS, ITEMS_PER_THREAD, THREADS_PER_BLOCK`. The default values should be 
fine, but the optimal value for `BLOCKS` is the number of `SMs` multiplied
by the occupancy of the `rough_partition` kernel if scheduling is optimal. 
If not, you may want a small multiple of this.

# Use case

This algorithm is meant for memory-constrained applications, as CUB's 
implementation is not in-place. For large, non-adversarial input there is no
significant performance penalty. For small arrays there is, and CUB's 
implementation is preferred.

# More Information

A research artefact with code to check for correctness and some benchmarks is
available at 
[partition-artefact](https://github.com/thomas3494/partition-artefact). 
The paper [not published yet]() explains the algorithm.
