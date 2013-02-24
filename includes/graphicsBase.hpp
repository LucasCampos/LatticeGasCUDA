/*
 * Class: Basic 2D drawing board
 * Author: Lucas Costa Campos
 * Date: 06/10/2012
 * 
 *
 * Class representing a 2D drawing canvas.
 * It uses GLFW, therefore OpenGL, to draw Drawables on
 * the screen. A vector of pointers to Drawables objects
 * is used on the function Draw
 *
 * It implements some default callbacks. To use local ones,
 * call the constructor with defaultCallbacks flag set to false
 *
 */
#ifndef GRAPHICS_BASE2D_HPP
#define GRAPHICS_BASE2D_HPP
#include <GL/glfw.h>
#include <vector>
#include <cstdlib>
#include <string>
#include <iostream>

namespace Graphics {

	class GraphicsBase2D {

		private:
			double timeBetweenFrames;

			static void GLFWCALL KeyCallback(int key, int action){
				if( action != GLFW_PRESS )
				{
					return;
				}

				switch (key)
				{
					case GLFW_KEY_ESC:
						Shutdown(0);
						break;
				}

			}

			static void WindowSizeCallback(int width, int height) {
				glViewport(0,0,width, height);
			}

			static void PrintVersion() {
				int major, minor, rev;
				glfwGetVersion(&major, &minor, &rev);
				std::cout << "You are using GLFW version " << major << "." << minor << "." << rev << std::endl;
			}


			static void Shutdown(int code) {
				glfwTerminate();
				exit(code);
			}

			void RegisterCallbacks() {
				glfwSetKeyCallback(KeyCallback);
				glfwSetWindowSizeCallback(WindowSizeCallback);
			}


		public:

			GraphicsBase2D() :timeBetweenFrames(0.001) {
				bool isOk = true;
				isOk |= glfwInit();
				isOk |= glfwOpenWindow( 600,600, 0,0,0,0,0,0, GLFW_WINDOW );
		
				if (isOk == false) {
					Shutdown(1);
					std::cout << "Erro starting GLFW." << std::endl;
				}

				RegisterCallbacks();

				glfwSetWindowTitle("GLFW Basic Window");
				glMatrixMode(GL_PROJECTION);
				glOrtho (-1, 1, -1, 1, 0, 1);

			}

			GraphicsBase2D(int windowWidth, int windowHeight, double boxMinX, double boxMaxX, double boxMinY, double boxMaxY, double _timeBetweenFrames, std::string windowName, bool defaultCallbacks) :timeBetweenFrames(_timeBetweenFrames) {
				bool isOk = true;
				isOk |= glfwInit();
				isOk |= glfwOpenWindow(windowWidth,windowHeight, 0,0,0,0,0,0, GLFW_WINDOW );
		
				if (!isOk) {
					Shutdown(1);
					std::cout << "Erro starting GLFW." << std::endl;
				}

				if (defaultCallbacks)
					RegisterCallbacks();

				glfwSetWindowTitle(windowName.c_str());
				glMatrixMode(GL_PROJECTION);
				glOrtho (boxMinX, boxMaxX, boxMinX, boxMaxY, 0, 1);

			}

			virtual ~GraphicsBase2D(){
				Shutdown(0);
			}

			void Clear() {
				glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
			}

			void EndFrame() {
				glfwSwapBuffers();
				glfwSleep(timeBetweenFrames);
			}

			template <class T>
			void Draw(const std::vector<T>& d){
				for (int i=0; i<d.size(); i++) 
					d[i]->Draw();
			}
	};
}
#endif
