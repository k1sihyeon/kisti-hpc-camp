nvcc -arch=sm_70 cg_driver.cu cg.cu -Xcompiler -fopenmp -o cg -lcusparse -lcusolver


