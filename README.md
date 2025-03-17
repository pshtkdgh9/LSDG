## LSDG

Recently, there has been active research on utilizing GPUs for the efficient processing of large-scale dynamic graphs. However, challenges arise due to the repeated transmission and processing of identical data during dynamic graph operations. This paper proposes an efficient processing scheme for large-scale dynamic graphs in GPU environments with limited memory, leveraging dynamic scheduling and operation reduction. The proposed scheme partitions the dynamic graph and schedules each partition based on active and tentative active vertices, optimizing GPU utilization. Additionally, snapshots are employed to capture graph changes, enabling the detection of redundant edge and vertex modifications. This reduces unnecessary computations, thereby minimizing GPU workloads and data transmission costs. The scheme significantly enhances performance by eliminating redundant operations on the same edges or vertices. Performance evaluations demonstrate an average improvement of 280% over existing static graph processing techniques and 108% over existing dynamic graph processing schemes.

#### Graph formats

EGraph accepts the binary serialized pre-built CSR graph representation. Reading binary formats is faster and more space efficient.

#### Compilation

To compile EGraph, just run make in the root directory. The only requrements are g++ and CUDA toolkit.

#### Running applications in LSDG

The applications take a graph as input as well as some optional arguments. For example:

```
$ ./sssp-TGP --input path-to-input-graph
$ ./sssp-TGP --input path-to-input-graph --source 10


#  ./sswp-TGP --input /home/netdb/jihyeon/make_snapshot/twitter7_snapshot4.el
