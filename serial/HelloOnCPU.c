#include <stdio.h>

void printhello(void) {
	printf("Hello World on CPU\n");
}

int main(void) {
	int i;
	for(i = 0; i < 4; i++) {
		printhello();
	}

	return 0;
}
