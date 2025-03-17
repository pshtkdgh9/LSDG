
CC=g++
NC=nvcc
CFLAGS=-std=c++11 -O3
NFLAGS=-arch=sm_50

SHARED=../shared

all:  sssp-TGP.o cc-TGP.o bfs-TGP.o sswp-TGP.o jihyeon.o
	
bfs-TGP.o: bfs-TGP.cu
	$(NC) -c bfs-TGP.cu $(CFLAGS) $(NFLAGS)		

cc-TGP.o: cc-TGP.cu
	$(NC) -c cc-TGP.cu $(CFLAGS) $(NFLAGS)
		
sssp-TGP.o: sssp-TGP.cu
	$(NC) -c sssp-TGP.cu $(CFLAGS) $(NFLAGS)		

sswp-TGP.o: sswp-TGP.cu
	$(NC) -c sswp-TGP.cu $(CFLAGS) $(NFLAGS)

jihyeon.o: jihyeon.cu
	$(NC) -c jihyeon.cu $(CFLAGS) $(NFLAGS)

clean:
	rm  *.o 
