LatticeGasCUDA
==============

Resultado do projeto de Métodos Computacionais Para a Física. Versão usando CUDA C. A versão paralela pode ser encontrada em https://github.com/LucasCampos/LatticeGa

==================REQUISITOS==================

Este programa requer o libglfw e libglew instalado. No Ubuntu, pode-se fazer

	sudo apt-get install libglfw-dev libglew1.6-dev

É necessário também ter CUDA instalado. No desenvolvimento do programa foi usado CUDA 5.0, mas versões mais antigas também devem funcionar bem.

==================COMPILAÇÃO==================

Para compilar este programa, é necessário usar somente

	make

Também existe a opção de compilar e rodar o programa

	make run

Se o seu computador não utilizar o chip Optimus, é necessário alterar a linha 28 de makefile para

	./$(EXECS)

==================MÉTRICA=====================

Velocidade de iteração, em um processador i7-3612QM e GTX630M, sem visualização

|L   | ms/ite |
|----|--------|
|192 | 0.05864 |
|384 | 0.22248 |
|576 | 0.48028 |
|768 | 0.84067 |
