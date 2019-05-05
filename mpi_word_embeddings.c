#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>	
#include <string.h>


#define MIN(a,b) (((a)<=(b))?(a):(b))

#define MAX_WORD_LENGTH 20
#define WORD_MATRIX_ROW_SIZE 10
#define WORD_MATRIX_COL_SIZE 300

// command and control protocol
const int COMMAND_EXIT =  0;
const int COMMAND_QUERY = 1;
const int CALCULATE_QUERY = 2;

#define FILENAME "word_embeddings_small.txt"

/*
	/// This method divides the matrix and words vector equally between processes. 
	/// For each slave process we create a words_to_send and matrix_to_send vectors. 
	/// Instead of reading the matrix and then dividing, we calculate the range that will be allocated for each process. 

	@param word_embedding_file File that was opened in main 
	@param my_rank this is actual rank for this case it will be 0. 
	@param process_count is the total amout of processes in MPI_COMM_WORLD
	@param c is the quotient in order to apply the formula to divide equally 
	@param r is the remainder in order to apply the formula to divide equally
*/

void distribute_word_matrix(FILE* word_embedding_file, int my_rank, int process_count,int c, int r)
{
	for (int num_process_destination = 1; num_process_destination < process_count; ++num_process_destination) 
	{
		int start = (num_process_destination-1) * c + MIN((long long)(num_process_destination-1), (long long)r);
		int final = (num_process_destination) * c + MIN((long long)(num_process_destination), (long long)r);
		int sub_process_range = abs(final-start);
		// printf ("%d: num_process_destination = %d has start %d and final %d\n", my_rank, num_process_destination, start, final);
		char * words_to_send = (char *) calloc (sub_process_range * MAX_WORD_LENGTH, sizeof (char));
		double * matrix_to_send = (double *) calloc (sub_process_range * WORD_MATRIX_COL_SIZE, sizeof (double));
        // First column is the word, not some data.
		for (int row_index = 0; row_index < sub_process_range; ++ row_index)
		{
			// Reads word 
			fscanf (word_embedding_file, "%s", &words_to_send [row_index * MAX_WORD_LENGTH]);
			for (int col_index = 0; col_index < WORD_MATRIX_COL_SIZE; ++ col_index)
			{
				// Reads the embeddings associated to the word
				fscanf (word_embedding_file, "%lf", &matrix_to_send [row_index * WORD_MATRIX_COL_SIZE + col_index]);
			}
		}
		printf ("%d: Sending words to process %d...\n", my_rank, num_process_destination);
		MPI_Send (words_to_send, sub_process_range * MAX_WORD_LENGTH, MPI_UNSIGNED_CHAR, num_process_destination, 0, MPI_COMM_WORLD);
		printf ("%d: Sending matrix to process %d...\n", my_rank, num_process_destination);
		MPI_Send (matrix_to_send, sub_process_range * WORD_MATRIX_COL_SIZE, MPI_DOUBLE, num_process_destination, 1, MPI_COMM_WORLD);
    	free (words_to_send);
    	free (matrix_to_send);
	}
}

void send_exit(int my_rank, int process_count)
{
	for(int process_num = 1; process_num < process_count; ++process_num)
	{
		printf("%d %s %d\n", my_rank, "Sending command", COMMAND_EXIT);
		MPI_Send(&COMMAND_EXIT, 1, MPI_INT, process_num, 2, MPI_COMM_WORLD); 
	}
}


void run_master_node(FILE* word_embedding_file, int my_rank, int process_count, int c, int r)
{
	char* queryWord = calloc(1024, sizeof(char));
	distribute_word_matrix(word_embedding_file, my_rank, process_count, c, r);
	while(0 == 0)
	{
		printf("Please type a query word:\n");
        scanf( "%s" , queryWord);
        if( strcmp(queryWord, "EXIT") == 0 )
        {
        	send_exit(my_rank, process_count); 
        	break; 
        }
        else
        {
        	for(int process_num = 1; process_num < process_count; ++process_num)
        	{
        		printf("%d %s %d\n", my_rank, "Sending command", COMMAND_QUERY);
        		MPI_Send(&COMMAND_QUERY, 1, MPI_INT, process_num, 3, MPI_COMM_WORLD);
        		MPI_Send(queryWord, 1024, MPI_CHAR, process_num, 4, MPI_COMM_WORLD); 
        	}
        	int target_process_id = -1; 
        	int word_index = -1; 
        	for(int process_num = 1; process_num < process_count; ++process_num)
        	{
        		MPI_Recv(&word_index, 1, MPI_INT, process_num, , MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        		if(word_index >= 0)
        			target_process_id = process_num;
        	}
        }
	}
	free(queryWord);
}

void run_slave_node(int my_rank, int c, int r)
{
    // Get quotient and remainder to use a formula to calculate start and final 
	int start = (my_rank-1) * c + MIN((long long)(my_rank-1), (long long)r);
	int final = (my_rank) * c + MIN((long long)(my_rank), (long long)r);
    int range = abs(final-start);

    // Create an array for storing the words 
    char* words = (char*) calloc(range * MAX_WORD_LENGTH, sizeof(char)); 

    // Create word embeddings matrix with contiguos memory 
    double* word_matrix = (double*) calloc(range * WORD_MATRIX_COL_SIZE, sizeof(double)); 

    printf("%d: %s\n", my_rank, "Receiving words...");
    MPI_Recv(words, range * MAX_WORD_LENGTH, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    printf("%d: %s\n", my_rank, "Receiving matrix...");
    MPI_Recv(word_matrix, range * WORD_MATRIX_COL_SIZE, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
	
	while(0 == 0)
	{
		int command = -1; 
		MPI_Recv(&command, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		printf("%d: %s %d\n", my_rank, "Receiving command", command);
		if(command == 0)
			break;
		else
		{

		}
	}


    free(words); 
    free(word_matrix);
}

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

	const int c = WORD_MATRIX_ROW_SIZE / (process_count-1);
	const int r = WORD_MATRIX_ROW_SIZE % (process_count-1);



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
		run_master_node(word_embedding_file, my_rank, process_count, c, r);
	}
	else
	{
		run_slave_node(my_rank, c, r);
	}

	// close file
	fclose(word_embedding_file); 

	MPI_Finalize(); 
	return 0;

}
