#ifndef LATTICE_HPP
#define LATTICE_HPP
#include "kernels.hpp"
#include <cmath>
#include <GL/glew.h>
#include <GL/glfw.h>
#include <cuda_gl_interop.h>
#include <vector>
#include <cstdio>
#include "helper.hpp"


void RetanguloCentrado(GLfloat* v, double xcentro, double ycentro, double b, double h) {

	v[0]=xcentro-0.5*b;
	v[1]=ycentro+0.5*h;
	v[2]=xcentro+0.5*b;
	v[3]=ycentro+0.5*h;
	v[4]=xcentro+0.5*b;
	v[5]=ycentro-0.5*h;
	v[6]=xcentro-0.5*b;
	v[7]=ycentro-0.5*h;

}

class Lattice {
	public:
		GLuint vboSquares;
		GLuint vboColors;
		GLuint vboArrows;
		struct cudaGraphicsResource* d_vboColors;
		struct cudaGraphicsResource* d_vboArrows;

		int L;
		int N;

		int* h_cells1; 
		int* h_cells2; 

		int* d_cells1; 
		int* d_cells2; 

		int* d_rules;

		float2* pos;
		float2* d_pos;

		int nBlocks;
		int nThreads;
		size_t memSize;

		Lattice(int _L=192, int threadMultiplier=1) {
			L=_L;
			N=L*L;

			memSize = sizeof(int)*N;

			h_cells1 = new int[N];
			h_cells2 = new int[N];
			pos = new float2[N];

			HANDLE_ERROR( cudaMalloc( (void **) &d_cells1, memSize ));
			HANDLE_ERROR( cudaMalloc( (void **) &d_cells2, memSize ));
			HANDLE_ERROR( cudaMalloc( (void **) &d_pos, sizeof(float2)*N ));
			HANDLE_ERROR( cudaMalloc( (void **) &d_rules, sizeof(int)*nRules ));
			SetRules();
			//std::cout << rules[STATIONARY] << std::endl;
			HANDLE_ERROR(cudaMemcpy(d_rules, rules, sizeof(rules), cudaMemcpyHostToDevice));
			cudaDeviceProp deviceProp;
			cudaGetDeviceProperties(&deviceProp, 0);
			nThreads=threadMultiplier*ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);
			nBlocks = N/nThreads + (N%nThreads == 0? 0:1);

			std::cout << "Numero de sitios: " << N << std::endl;
			std::cout << "Blocks: " << nBlocks << std::endl;
			std::cout << "Threads per Block: " << nThreads << std::endl;

			MakeTriangularLatticePos();

		}
		void PrepareToOpenGL() {
			glewInit();
			cudaGLSetGLDevice(0);

			MakeSquaresVBO();
			MakeColorVBO();
			MakeArrowVBO();

		}
		~Lattice() {
		    delete[] pos;
		    delete[] h_cells2;
		    delete[] h_cells1;
		    cudaFree(d_cells2);
		    cudaFree(d_cells1);
		    cudaFree(d_rules);
		}

		void CopyToDevice() {

			//Here copying to memory does not work
			HANDLE_ERROR(cudaMemcpy(d_cells1, h_cells1, memSize, cudaMemcpyHostToDevice));
			HANDLE_ERROR(cudaMemcpy(d_cells2, h_cells1, memSize, cudaMemcpyHostToDevice));
			HANDLE_ERROR(cudaMemcpy(d_pos, pos, N*sizeof(float2), cudaMemcpyHostToDevice));
		}

		void CopyToHost() {
			//Nor here
			HANDLE_ERROR(cudaMemcpy( h_cells1, d_cells1, memSize, cudaMemcpyDeviceToHost )); //Copia o vetor posição para o host.
		}

		static double uniforme() {return (rand()/(double)RAND_MAX);}

		void MakeArrowVBO() {

			const int Narrows = N*4;
			GLfloat p[Narrows];
			for (int i=0; i<Narrows; i++) {
				p[i] = 0;
			}

			vboArrows = createVBO(p, sizeof(p), GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
			cudaGraphicsGLRegisterBuffer(&d_vboArrows, vboArrows, cudaGraphicsMapFlagsWriteDiscard);
		}

		void MakeColorVBO() {

			const int Ncolors = N*3*4;
			GLfloat colors[Ncolors];
			for (int i=0; i<Ncolors; i+=24) {
				double c1= uniforme();
				double c2= uniforme();
				double c3= uniforme();
				for (int j=0; j<24; j+=3) {
					colors[i+j] = c1;
					colors[i+j+1] = c2;
					colors[i+j+2] = c3;
				}
			}
			vboColors = createVBO(colors, sizeof(colors), GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
			cudaGraphicsGLRegisterBuffer(&d_vboColors, vboColors, cudaGraphicsMapFlagsWriteDiscard);
		}

		void MakeSquaresVBO() {

			GLfloat squares[N*8];
			const double size = 1.0/L;

			for (int i=0; i<(N*8); i+=8) {
				RetanguloCentrado(&squares[i],pos[i/8].x,pos[i/8].y,size,size);
				//std::cout << pos[i/8].x << "  " << pos[i/8].y << std::endl;
			}
			vboSquares = createVBO(squares, sizeof(squares), GL_ARRAY_BUFFER, GL_STREAM_DRAW);
		}

		void UpdateColor() {
			float* d_color;
			size_t num_bytes;
			cudaGraphicsMapResources(1, &d_vboColors, 0);
			cudaGraphicsResourceGetMappedPointer((void**)&d_color, &num_bytes, d_vboColors);
			Density <<< nBlocks, nThreads>>>(d_cells1, d_color,N);
			cudaGraphicsUnmapResources(1, &d_vboColors, 0);
		}

		void UpdateMomenta() {
			float* d_arrows;
			size_t num_bytes;
			cudaGraphicsMapResources(1, &d_vboArrows, 0);
			cudaGraphicsResourceGetMappedPointer((void**)&d_arrows, &num_bytes, d_vboArrows);
			Momenta <<< nBlocks, nThreads >>> (d_cells1, d_pos, d_arrows, L, N);
			cudaGraphicsUnmapResources(1, &d_vboArrows, 0);
		}

		void DrawArrows() {
			UpdateMomenta();
			UpdateColor();
			//glBindBufferARB(GL_ARRAY_BUFFER_ARB, vboColors); 
			//glEnableClientState(GL_COLOR_ARRAY);             // activate vertex coords array
			//glColorPointer(3, GL_FLOAT, 0, 0);

			glColor3f(1.0,1.0,1.0);

			glBindBufferARB(GL_ARRAY_BUFFER_ARB, vboArrows);
			glEnableClientState(GL_VERTEX_ARRAY);
			glVertexPointer(2, GL_FLOAT, 0, 0);
			glDrawArrays(GL_LINES, 0,N*2);

			//glDisableClientState(GL_COLOR_ARRAY);             // activate vertex coords array
			glDisableClientState(GL_VERTEX_ARRAY);
			DrawBarrier();
		}

		void Draw() {
			UpdateColor();
			glBindBufferARB(GL_ARRAY_BUFFER_ARB, vboColors); 
			glEnableClientState(GL_COLOR_ARRAY);             // activate vertex coords array
			glColorPointer(3, GL_FLOAT, 0, 0);

			glBindBufferARB(GL_ARRAY_BUFFER_ARB, vboSquares);
			glEnableClientState(GL_VERTEX_ARRAY);
			glVertexPointer(2, GL_FLOAT, 0, 0);
			glDrawArrays(GL_QUADS, 0,N*4);

			glDisableClientState(GL_COLOR_ARRAY);             // activate vertex coords array
			glDisableClientState(GL_VERTEX_ARRAY);


		}

		void DrawBarrier()  const {
			const double size = 1.0/L;
			glColor3f(1.0,0.0,0);
			glBegin(GL_QUADS);
			for (int i=0; i<N; i++) {
				if ((h_cells1[i]&BARRIER) == BARRIER){
					glVertex2d(pos[i].x-0.5*size, pos[i].y+0.5*size);
					glVertex2d(pos[i].x+0.5*size, pos[i].y+0.5*size);
					glVertex2d(pos[i].x+0.5*size, pos[i].y-0.5*size);
					glVertex2d(pos[i].x-0.5*size, pos[i].y-0.5*size);
				}
			}
			glEnd();
		}

		void MakeTriangularLatticePos() {
			int n =L;
			double ybox=1;

			double dx=0, dy=0, del = ybox/n;

			int counter =0;
			double dxbase = 0;

			while (counter < N) {

				for (int i=0; i<n; i++) {
					pos[counter].x = dx;
					pos[counter].y = dy;
					dx+=del;
					counter++;
				}
				dy += del*sqrt(3.0)/2;
				dxbase += del/2;
				dx = dxbase;
			}

			for (int i=0; i<N; i++){
				if (pos[i].x > 1) pos[i].x -= 1;
				if (pos[i].y > 1) pos[i].y -= 1;

			}
		};


		void UpdateLattice1() {
			UpdateCells <<< nBlocks, nThreads>>>(d_cells1, d_cells2, d_pos, d_rules, L, N);
		}

		void UpdateLattice2() {
			UpdateCells <<< nBlocks, nThreads>>>(d_cells2, d_cells1, d_pos, d_rules, L, N);
		}
		void FullUpdate() {
			//std::cout << "here" << std::endl;
			UpdateLattice2();
			UpdateLattice1();
		}

		friend std::ostream& operator<<(std::ostream& stream,const Lattice& c){


			for (int i=0; i<c.N; i++) {
				stream << c.h_cells1[i] << std::endl;
			}
			return stream;
		}
};

#endif
