#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>	


#define MIN(a,b) (((a)<=(b))?(a):(b))


int main(int argc, char *argv[])
{
	MPI_Init(&argc, &argv);

	int my_rank = -1; 
	int process_count = -1;

	// get my_rank
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	// get process_count 
	MPI_Comm_size(MPI_COMM_WORLD, &process_count);

	int word_matrix_row_size = 10;  
	int word_matrix_col_size = 300; 

	// Get quotient and remainder to use a formula to calculate start and final 
	int c = word_matrix_row_size / process_count;
	int r = word_matrix_row_size % process_count;
	int start = my_rank * c + MIN((long long)my_rank, (long long)r);
	int final = (my_rank+1) * c + MIN((long long)(my_rank+1), (long long)r);
	int range = final-start; 


	// Open file, change this for your working directory. 
	FILE* word_embedding_file = fopen("/home/lesan2807/Documents/MPIImplementation/Untitled Folder/word-embedding/word_embeddings_small.txt", "r");
	if( !word_embedding_file)
	{
		printf("%s\n", "Could not open file. Ending program");
		MPI_Finalize(); 
		return 1; 
	}

	printf("[%d-%d[: %d\n", start, final, range);
	// Create an array for storing the words 
	char* words = (char*)malloc((final-start ) * sizeof(char)); 

	// Create word embeddings matrix 
	// float** word_matrix = (float**)malloc( range * sizeof(float*));
	// for ( int row_index = 0; row_index <	 word_matrix_row_size; ++row_index )
	// 	word_matrix[row_index] = (float*)malloc(word_matrix_col_size * sizeof(float));
	float** word_matrix = calloc(range, sizeof(float*)); 
	for( int row_index = 0; row_index < range; ++row_index)
		word_matrix[row_index] = calloc(word_matrix_col_size, sizeof(float));	


	if(my_rank == 0)
	{
		int count_words = 0; 
		for(int row_index = 0; row_index < word_matrix_row_size; ++row_index) // TODO: THIS PRODUCES SEgFAULT FIX IT
		{
			for(int col_index = 0; col_index < word_matrix_col_size; ++col_index)
			{
				if(col_index == 0)
				{
					fscanf(word_embedding_file, "%s", words+count_words); 
					//printf("%s\t", words+count_words);
					++count_words; 
				}
				fscanf(word_embedding_file, "%f", &word_matrix[row_index][col_index]); 
				//printf("%0.16f\t", word_matrix[row_index][col_index]);
			}
			//printf("\n");
		}
	}



	// free memory 
	free(words); 

	for ( int row_index = 0; row_index < range; ++row_index )
		free(word_matrix[row_index]);
	free(word_matrix);

	// close file
	fclose(word_embedding_file); 

	MPI_Finalize(); 
	return 0;
	
}