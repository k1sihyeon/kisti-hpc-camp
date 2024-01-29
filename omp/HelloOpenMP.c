#include <stdio.h>
#include <omp.h>

void printhello(void) {
	printf("Hello World on CPU!\n");
}

int main(void) {
	#pragma omp parallel
	{
		printhello();
	}

	return 0;
}

