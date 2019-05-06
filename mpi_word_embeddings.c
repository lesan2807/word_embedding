#include <mpi.h>
#include <stdio.h>
#include <stdlib.h> 
#include <string.h>


#define MIN(a,b) (((a)<=(b))?(a):(b))

#define MAX_WORD_LENGTH 20
#define WORD_MATRIX_ROW_SIZE 10
#define WORD_MATRIX_COL_SIZE 300

// Message tags.
#define TAG_SEND_WORDS 0
#define TAG_SEND_MATRIX 1
#define TAG_SEND_COMMAND 2
#define TAG_SEND_QUERY 3
#define TAG_SEND_WORD_INDEX 4
#define TAG_SEND_WORD_EMBEDDINGS 5

// Command and control protocol.
const int COMMAND_EXIT = 0;
const int COMMAND_QUERY = 1;
const int COMMAND_CALCULATE_SIMILARITY = 2;

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
    int start = (num_process_destination-1) * c + MIN ((long long)(num_process_destination-1), (long long)r);
    int final = (num_process_destination) * c + MIN ((long long)(num_process_destination), (long long)r);
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
    MPI_Send (words_to_send, sub_process_range * MAX_WORD_LENGTH, MPI_UNSIGNED_CHAR, num_process_destination, TAG_SEND_WORDS, MPI_COMM_WORLD);
    printf ("%d: Sending matrix to process %d...\n", my_rank, num_process_destination);
    MPI_Send (matrix_to_send, sub_process_range * WORD_MATRIX_COL_SIZE, MPI_DOUBLE, num_process_destination, TAG_SEND_MATRIX, MPI_COMM_WORLD);
      free (words_to_send);
      free (matrix_to_send);
  }
}


void run_master_node (FILE* word_embedding_file, int my_rank, int process_count, int c, int r)
{
    distribute_word_matrix (word_embedding_file, my_rank, process_count, c, r);
    while (1)
    {
        printf ("Please type a query word:\n");

        char query_word [1024];
        scanf ("%1023s" , query_word);
        if (strcmp(query_word, "EXIT") == 0)
        {
            for (int process_num = 1; process_num < process_count; ++process_num)
            {
                printf ("%d %s %d\n", my_rank, "Sending command", COMMAND_EXIT);
                MPI_Send (&COMMAND_EXIT, 1, MPI_INT, process_num, TAG_SEND_COMMAND, MPI_COMM_WORLD); 
            }
            break; 
        } else { // Got a word to query instead of EXIT.
            for (int process_num = 1; process_num < process_count; ++process_num)
            {
                printf ("%d: %s %d %s\n", my_rank, "Sending command and query", COMMAND_QUERY, query_word);
                MPI_Send (&COMMAND_QUERY, 1, MPI_INT, process_num, TAG_SEND_COMMAND, MPI_COMM_WORLD);
                // TODO: Change into broadcast.
                MPI_Send (query_word, strlen (query_word) + 1 /* include \0 */, MPI_CHAR, process_num, TAG_SEND_QUERY, MPI_COMM_WORLD); 
            }
            printf ("%d: Receiving word indexes...\n", my_rank);
            int found = 0; // boolean.
            double word_embeddings [WORD_MATRIX_COL_SIZE];
            for (int process_num = 1; process_num < process_count; ++process_num) {
                int received_index = -2;
                MPI_Recv (&received_index, 1, MPI_INT, process_num, TAG_SEND_WORD_INDEX, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                printf ("%d: Received index %d from process %d\n", my_rank, received_index, process_num);
                if (received_index != -1) {
                    printf ("%d: Found word on process %d\n", my_rank, process_num);
                    MPI_Recv (word_embeddings, WORD_MATRIX_COL_SIZE, MPI_DOUBLE, process_num, TAG_SEND_WORD_EMBEDDINGS, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    printf ("%d: Received word embeddings, first val is %lf\n", my_rank, word_embeddings [0]);
                    found = 1;
                    // break; // Don't want unread messages.
                }
            }
            if (found) {
                // TODO: Change into broadcast.
                for (int process_num = 1; process_num < process_count; ++ process_num) {
                    printf ("%d: Sending command to calculate similarity to process %d\n", my_rank, process_num);
                    MPI_Send (&COMMAND_CALCULATE_SIMILARITY, 1, MPI_INT, process_num, TAG_SEND_COMMAND, MPI_COMM_WORLD);
                    printf ("%d: Sending word embeddings to process %d\n", my_rank, process_num);
                    MPI_Send (word_embeddings, WORD_MATRIX_COL_SIZE, MPI_DOUBLE, process_num, TAG_SEND_WORD_EMBEDDINGS, MPI_COMM_WORLD);
                }
            } else {
                printf ("Query word was not found\n");
            }
        }
    }
}

int find_word_index (int my_rank, char* words, char* query_word, int range, double * results_to_fill)
{
    for (int index = 0; index < range; ++index)
    {
        printf ("%d: %s\n", my_rank, &words[index*MAX_WORD_LENGTH]);
        if (strcmp (&words[index*MAX_WORD_LENGTH], query_word) == 0)
            return index; 
    }
    return -1;
}

void run_slave_node (int my_rank, int c, int r)
{
    // Get quotient and remainder to use a formula to calculate start and final 
  int start = (my_rank-1) * c + MIN ((long long)(my_rank-1), (long long)r);
  int final = (my_rank) * c + MIN ((long long)(my_rank), (long long)r);
    int range = abs (final-start);

    // Create an array for storing the words 
    char* words = (char*) calloc(range * MAX_WORD_LENGTH, sizeof(char)); 

    // Create word embeddings matrix with contiguos memory 
    double* word_matrix = (double*) calloc (range * WORD_MATRIX_COL_SIZE, sizeof(double)); 

    printf("%d: %s\n", my_rank, "Receiving words...");
    MPI_Recv(words, range * MAX_WORD_LENGTH, MPI_UNSIGNED_CHAR, 0, TAG_SEND_WORDS, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    printf("%d: %s\n", my_rank, "Receiving matrix...");
    MPI_Recv(word_matrix, range * WORD_MATRIX_COL_SIZE, MPI_DOUBLE, 0, TAG_SEND_MATRIX, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
  
    while(1)
    {
        int command = -1; 
        MPI_Recv (&command, 1, MPI_INT, 0, TAG_SEND_COMMAND, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf ("%d: %s %d\n", my_rank, "Received command", command);
        if (command == COMMAND_EXIT)
            break;
        else if (command == COMMAND_QUERY)
        {
            char query_word [1024];
            double query_weights [WORD_MATRIX_COL_SIZE];
            MPI_Recv (query_word, 1024, MPI_CHAR, 0, TAG_SEND_QUERY, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
            printf ("%d: Received query word %s\n", my_rank, query_word);
            int word_index = find_word_index (my_rank, words, query_word, range, query_weights);
            printf ("%d: Sending index %d\n", my_rank, word_index);
            MPI_Send (&word_index, 1, MPI_INT, 0, TAG_SEND_WORD_INDEX, MPI_COMM_WORLD);
            if (word_index != -1) {
                printf ("%d: Sending word embeddings...\n", my_rank);
                MPI_Send (query_weights, WORD_MATRIX_COL_SIZE, MPI_DOUBLE, 0, TAG_SEND_WORD_EMBEDDINGS, MPI_COMM_WORLD);
            }
        }
        else if (command == COMMAND_CALCULATE_SIMILARITY) {
            printf ("%d Received calculate similarity command\n", my_rank);
            double word_embeddings [WORD_MATRIX_COL_SIZE];
            MPI_Recv (word_embeddings, WORD_MATRIX_COL_SIZE, MPI_DOUBLE, 0, TAG_SEND_WORD_EMBEDDINGS, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            printf ("%d TODO: COMMAND_CALCULATE_SIMILARITY\n", my_rank);
        }
    }

    free (words); 
    free (word_matrix);
}

int main(int argc, char *argv[])
{
  MPI_Init(&argc, &argv);

  int my_rank = -1; 
  int process_count = -1;

  // get my_rank
  MPI_Comm_rank (MPI_COMM_WORLD, &my_rank);
  // get process_count 
  MPI_Comm_size (MPI_COMM_WORLD, &process_count);

  if (process_count < 3)
  {
      if (my_rank == 0)
          printf ("%s\n", "For this program we want more than 1 slave process");
      MPI_Finalize (); 
      return 1; 
  }

  // Open file.
  FILE* word_embedding_file = fopen (FILENAME, "r");
  if (!word_embedding_file)
  {
      printf ("%s\n", "Could not open file. Ending program");
      MPI_Finalize (); 
      return 2; 
  }

  const int c = WORD_MATRIX_ROW_SIZE / (process_count-1);
  const int r = WORD_MATRIX_ROW_SIZE % (process_count-1);
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
