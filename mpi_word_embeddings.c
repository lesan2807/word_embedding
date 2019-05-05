#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>	


#define MIN(a,b) (((a)<=(b))?(a):(b))

#define MAX_WORD_LENGTH 10

// change this for your directory
#define FILENAME "word_embeddings_small.txt"

int main(int argc, char *argv[])
{
	MPI_Init(&argc, &argv);

	int my_rank = -1; 
	int process_count = -1;

	// get my_rank
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	// get process_count 
	MPI_Comm_size(MPI_COMM_WORLD, &process_count);

	if(process_count < 3)
	{
		if(my_rank == 0)
			printf("%s\n", "For this program we want more than 1 slave process");
		MPI_Finalize(); 
		return 1; 
	}

	int word_matrix_row_size = 10;  
	int word_matrix_col_size = 300; 

	// Get quotient and remainder to use a formula to calculate start and final 
	int c = word_matrix_row_size / (process_count-1);
	int r = word_matrix_row_size % (process_count-1);
	int start = (my_rank-1) * c + MIN((long long)(my_rank-1), (long long)r);
	int final = (my_rank) * c + MIN((long long)(my_rank), (long long)r);
	int range = abs(final-start); // rank 0 has to have a size 

	//printf("Range: [%d, %d[ from process %d\n", start, final, my_rank);

	// Create an array for storing the words 
	char* words = (char*) calloc(word_matrix_row_size * MAX_WORD_LENGTH, sizeof(char)); 

	// Create word embeddings matrix with contiguos memory 
	double* word_matrix = (double*) calloc(word_matrix_row_size * word_matrix_col_size, sizeof(double)); 


	// Open file,
	FILE* word_embedding_file = fopen(FILENAME, "r");
	if( !word_embedding_file)
	{
		printf("%s\n", "Could not open file. Ending program");
		MPI_Finalize(); 
		return 2; 
	}


	if(my_rank == 0) // Process 0 reads all the matrix sends sub matrix to others 
	{
		int num_process_destination = 1; 
		for(int row_index = 0; row_index <= word_matrix_row_size; ++row_index) 
		{
			int distribute_index = row_index % (range - 1); 
			for(int col_index = 0; col_index < word_matrix_col_size; ++col_index)
			{
				if(col_index == 0)
				{
					fscanf(word_embedding_file, "%s", &words[row_index * MAX_WORD_LENGTH + col_index]); 
				}
				fscanf(word_embedding_file, "%lf", &word_matrix[distribute_index * word_matrix_col_size + col_index]); 
				//printf("%0.16f\t", word_matrix[distribute_index * word_matrix_col_size + col_index]);
			}
			if(distribute_index == 0 )
			{
				if(num_process_destination < process_count)
				{
					//printf("Sending to %d...\n", num_process_destination);
					MPI_Send(words, word_matrix_row_size * MAX_WORD_LENGTH , MPI_UNSIGNED_CHAR, num_process_destination, 0, MPI_COMM_WORLD);
					MPI_Send(word_matrix,  (range - 1) * word_matrix_col_size, MPI_DOUBLE, num_process_destination, 1, MPI_COMM_WORLD); 
					num_process_destination++;  
				}
			}
			printf("%d\n", distribute_index);
		}
	}
	else
	{
		MPI_Recv(words, word_matrix_row_size * MAX_WORD_LENGTH, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(word_matrix, range * word_matrix_col_size, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 

		printf("%s%d\n", "I am process #", my_rank);

	// 	for(int index = 0; index < range; ++index)
	// 	{
	// 		for(int jndex = 0; jndex < word_matrix_col_size; ++jndex)
	// 		{
	// 			printf("[%lf]", word_matrix[index * range + jndex]);
	// 		}
	// 		printf("%s%d\n", "from process#", my_rank);
	// 	}
	 }


	free(words); 
	free(word_matrix);

	// close file
	fclose(word_embedding_file); 

	MPI_Finalize(); 
	return 0;
	
}