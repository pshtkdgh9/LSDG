

CC=g++
NC=nvcc
CFLAGS=-std=c++11 -O3
NFLAGS=-arch=sm_50

SHARED=shared
EGRAPH=egraph
TOOLS=tools

DEP=$(SHARED)/timer.o $(SHARED)/argument_parsing.o $(SHARED)/graph.o $(SHARED)/subgraph.o $(SHARED)/partitioner.o $(SHARED)/subgraph_generator.o $(SHARED)/gpu_kernels.o $(SHARED)/egraph_utilities.o  #$(SHARED)/test.o  

all: make1 make2  sssp-TGP cc-TGP bfs-TGP sswp-TGP jihyeon # bfs-sync cc-sync sssp-sync sswp-sync bfs-async 

make1:
	make -C $(SHARED)

make2:
	make -C $(EGRAPH)


# bfs-sync: $(EGRAPH)/bfs-sync.o $(DEP)
# 	$(NC) $(EGRAPH)/bfs-sync.o $(DEP) -o bfs-sync $(CFLAGS) $(NFLAGS)
	
# cc-sync: $(EGRAPH)/cc-sync.o $(DEP)
# 	$(NC) $(EGRAPH)/cc-sync.o $(DEP) -o cc-sync $(CFLAGS) $(NFLAGS)

# sssp-sync: $(EGRAPH)/sssp-sync.o $(DEP)
# 	$(NC) $(EGRAPH)/sssp-sync.o $(DEP) -o sssp-sync $(CFLAGS) $(NFLAGS)

# sswp-sync: $(EGRAPH)/sswp-sync.o $(DEP)
# 	$(NC) $(EGRAPH)/sswp-sync.o $(DEP) -o sswp-sync $(CFLAGS) $(NFLAGS)

bfs-TGP: $(EGRAPH)/bfs-TGP.o $(DEP)
	$(NC) $(EGRAPH)/bfs-TGP.o $(DEP) -o bfs-TGP $(CFLAGS) $(NFLAGS)	
	
cc-TGP: $(EGRAPH)/cc-TGP.o $(DEP)
	$(NC) $(EGRAPH)/cc-TGP.o $(DEP) -o cc-TGP $(CFLAGS) $(NFLAGS)		
	
sssp-TGP: $(EGRAPH)/sssp-TGP.o $(DEP)
	$(NC) $(EGRAPH)/sssp-TGP.o $(DEP) -o sssp-TGP $(CFLAGS) $(NFLAGS)	

sswp-TGP: $(EGRAPH)/sswp-TGP.o $(DEP)
	$(NC) $(EGRAPH)/sswp-TGP.o $(DEP) -o sswp-TGP $(CFLAGS) $(NFLAGS)	

jihyeon: $(EGRAPH)/jihyeon.o $(DEP)
	$(NC) $(EGRAPH)/jihyeon.o $(DEP) -o jihyeon $(CFLAGS) $(NFLAGS)	
	
clean:
	make -C $(SHARED) clean
	make -C $(EGRAPH) clean
	#make -C $(TOOLS) clean
	rm -f bfs-TGP cc-TGP sssp-TGP sswp-TGP jihyeon
