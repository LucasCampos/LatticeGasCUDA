#include <iostream>
#include <cstdlib>
#include "rules.hpp"
#include "kernels.hpp"
#include "lattice.hpp"
#include "graphicsBase.hpp"
#include <fstream>

int UniformBetween(int nMax) {return rand()%nMax;}
using namespace std;

void MakeCircle(Lattice& lat, double xC, double yC, double radius, unsigned state) {

	for (int i=0; i<lat.N; i++) {
		double dx = lat.pos[i].x - xC;
		double dy = lat.pos[i].y - yC;
		double r = sqrt(dx*dx+dy*dy);

		if ( r < radius){
			lat.h_cells1[i] = state;
		}

	}

}

void InitAll(Lattice& lat, unsigned s) {
	for (int i=0; i<lat.N; i++) 
		lat.h_cells1[i] = s;
}

void swp(double& a, double& b) {
	double c = a;
	a=b;
	b=c;
}

void MakeRec(Lattice& lat, double x0, double y0, double xF, double yF, unsigned s) {

	if (x0>xF) swp(x0,xF);
	if (y0>yF) swp(y0,yF);

	for (int i=0; i<lat.N; i++){
		if ((lat.pos[i].x > x0) && (lat.pos[i].x < xF))
			if ((lat.pos[i].y > y0) && (lat.pos[i].y < yF)){
					lat.h_cells1[i] = s;
			}
	}

}

int main() {
	bool running = true;
	double factorTriang = sqrt(3.0)/2;
	srand(time(NULL));
	Graphics::GraphicsBase2D graphics(800*sqrt(3.0)/2, 800, 
			0, 1.0, 
			0, sqrt(3.0)/2, 
			0.01, "Gas LatticeCUDA", true);
	Lattice lat(192*2,16);

	InitAll(lat,RIGHT);
	MakeRec(lat, .3, .3*factorTriang, .32, .7*factorTriang, BARRIER);

	lat.PrepareToOpenGL();
	lat.CopyToDevice();
	glewInit();

	while(true){
		graphics.Clear();
		lat.DrawArrows();
		if (glfwGetKey('P') == GLFW_PRESS)
			running=!running;
		if (running)
			for (int i=0; i<5;i++)
				lat.FullUpdate();

		graphics.EndFrame();
		cudaThreadSynchronize();
	}
}
