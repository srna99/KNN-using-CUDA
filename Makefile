todo: cuda
cuda: cuda.cu
	nvcc -O3 -gencode arch=compute_61,code=sm_61 cuda.cu -o cuda -Ilibarff libarff/arff_attr.cpp libarff/arff_data.cpp libarff/arff_instance.cpp libarff/arff_lexer.cpp libarff/arff_parser.cpp libarff/arff_scanner.cpp libarff/arff_token.cpp libarff/arff_utils.cpp libarff/arff_value.cpp
clean:
	rm cuda
