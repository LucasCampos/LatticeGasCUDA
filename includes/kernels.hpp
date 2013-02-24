#ifndef CELL_HPP
#define CELL_HPP
#include <GL/glew.h>
#include <cstdio>

__device__ __host__ inline int RightNeigh    (int i, int L, int N) { 
	if ((i+1)%L == 0) i-= L;
	return i+1;
}
	
__device__ __host__ inline int LeftNeigh     (int i, int L,int N) { 
	if (i%L == 0) i+= L;
	return i-1;
}

__device__ __host__ inline int RightUpNeigh  (int i, int L, int N) {
	i+=L;
	if (i>=N) i-=N;
	return i;
}


__device__ __host__ inline int RightDownNeigh(int i, int L, int N) { 
	i-= (L-1);
	if (i<0) i+= N;
	return i;
}

__device__ __host__ inline int LeftUpNeigh (int i, int L, int N) { 
	i+= (L-1);
	if (i>=N) i-=N;
	return i;
}

__device__ __host__ inline int LeftDownNeigh   (int i, int L, int N) {
	i-=L;
	if (i<0) i+=N;
	return i;
}

__global__ void UpdateCells(int* inCells, int* outCells, float2* pos, int* rules, int L, int N) {
	const int i=blockIdx.x*blockDim.x+threadIdx.x;
	if (i >= N) {
		return;
	}
	int newCurr = inCells[i] & (BARRIER|STATIONARY);
	newCurr |= inCells[RightNeigh(i,L,N)] & LEFT;
	newCurr |= inCells[RightDownNeigh(i,L,N)] & LEFT_UP;
	newCurr |= inCells[LeftDownNeigh(i,L,N)] & RIGHT_UP;

	newCurr |= inCells[LeftNeigh(i,L,N)] & RIGHT;
	newCurr |= inCells[LeftUpNeigh(i,L,N)] & RIGHT_DOWN;
	newCurr |= inCells[RightUpNeigh(i,L,N)] & LEFT_DOWN;
	
	outCells[i] = rules[newCurr];
	if (pos[i].x < 0.01) outCells[i] = RIGHT;
	else if (pos[i].x > .99) outCells[i] = RIGHT;
	
}

__device__ __host__ int CountMass(int cell) {
	int mass=0;
	int curr = cell;
	while (curr > 0) {
		mass += curr&1;
		curr = curr >> 1;
	}
	return mass;
}

__global__ void GetCellsMass(int* cells, int* cellMass, int N) {
	const int i=blockIdx.x*blockDim.x+threadIdx.x;
	cellMass[i] = CountMass(cells[i]);
}

__global__ void Density(int* cells, GLfloat* color, int N) {
	const int i=blockIdx.x*blockDim.x+threadIdx.x;
	if (i >= N) {
		return;
	}
	const int iC=i*12;
	int curr = cells[i];
	if((curr & BARRIER) != 0)  {
		for (int j=0; j<12; j+=3) {
			color[iC+j] = 0;
			color[iC+j+1] = 0;
			color[iC+j+2] = .3;
		}
		return;
	}
	float den=CountMass(curr);

	den/=8;

	for (int j=0; j<12; j+=3) {
		color[iC+j] = den;
		color[iC+j+1] = 0;
		color[iC+j+2] = 0;
	}
}

__device__ __host__ int isThere(int current, int cond) {
	return ((current&cond)/cond);
}

__device__ __host__ float2 SingleMomentum(int current) {

	float2 momentum = (float2){0,0};

	momentum.x +=  isThere(current,RIGHT);
	momentum.x += -isThere(current,LEFT);

	momentum.x +=  cos(M_PI/3.0)*isThere(current,RIGHT_UP);
	momentum.x += -cos(M_PI/3.0)*isThere(current,LEFT_UP);

	momentum.x +=  cos(M_PI/3.0)*isThere(current,RIGHT_DOWN);
	momentum.x += -cos(M_PI/3.0)*isThere(current,LEFT_DOWN);


	momentum.y += sin(M_PI/3.0)*isThere(current,RIGHT_UP);
	momentum.y += sin(M_PI/3.0)*isThere(current,LEFT_UP);

	momentum.y += -sin(M_PI/3.0)*isThere(current,RIGHT_DOWN);
	momentum.y += -sin(M_PI/3.0)*isThere(current,LEFT_DOWN);

	return momentum;


}

__global__ void Momenta(int* cells, float2* pos, GLfloat* arrows, int L, int N) {
	const int i=blockIdx.x*blockDim.x+threadIdx.x;
	if (i >= N) {
		return;
	}
	int curr = cells[i];
	float2 momentum = SingleMomentum(curr);
	momentum.x /= L;
	momentum.y /= L;
	float2 posL = pos[i];

	arrows[i*4] = posL.x;
	arrows[i*4+1] = posL.y;
	arrows[i*4+2] = posL.x + momentum.x;
	arrows[i*4+3] = posL.y + momentum.y;

}

#endif
