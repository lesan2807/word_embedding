// NOTE: P is assumed to be process_count - 1.

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h> 
#include <string.h>
#include "omp.h"


#define MIN(a,b) (((a)<=(b))?(a):(b))

#define VERSION_TOP_P

#define MAX_WORD_LENGTH 20
#define WORD_MATRIX_ROW_SIZE 1000
#define WORD_MATRIX_COL_SIZE 300

// Message tags.
#define TAG_SEND_WORDS 0
#define TAG_SEND_MATRIX 1
#define TAG_SEND_COMMAND 2
#define TAG_SEND_QUERY 3
#define TAG_SEND_WORD_INDEX 4
#define TAG_SEND_WORD_EMBEDDINGS 5
#define TAG_SEND_WORD 6
#define TAG_SEND_SCORE 7

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
  for (int num_process_destination = 1; num_process_destination < process_count; ++ num_process_destination) 
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
        double read_val = -1;
        fscanf (word_embedding_file, "%lf", &read_val);
        matrix_to_send [row_index * WORD_MATRIX_COL_SIZE + col_index] = read_val;
      }
    }
    /*
    for (int i = 0; i < WORD_MATRIX_COL_SIZE; ++i) {
        printf (" %lf ", matrix_to_send [i]);
    }
    printf ("\n\n\n"); */
    //printf ("%d: Sending words to process %d...\n", my_rank, num_process_destination);
    MPI_Send (words_to_send, sub_process_range * MAX_WORD_LENGTH, MPI_UNSIGNED_CHAR, num_process_destination, TAG_SEND_WORDS, MPI_COMM_WORLD);
    //printf ("%d: Sending matrix to process %d...\n", my_rank, num_process_destination);
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
            #pragma omp for
            for (int process_num = 1; process_num < process_count; ++ process_num)
            {
                //printf ("%d %s %d\n", my_rank, "Sending command", COMMAND_EXIT);
                MPI_Send (&COMMAND_EXIT, 1, MPI_INT, process_num, TAG_SEND_COMMAND, MPI_COMM_WORLD); 
            }
            break; 
        } else { // Got a word to query instead of EXIT.
            #pragma omp for
            for (int process_num = 1; process_num < process_count; ++ process_num)
            {
                //printf ("%d: %s %d %s\n", my_rank, "Sending command and query", COMMAND_QUERY, query_word);
                MPI_Send (&COMMAND_QUERY, 1, MPI_INT, process_num, TAG_SEND_COMMAND, MPI_COMM_WORLD);
                MPI_Send (query_word, strlen (query_word) + 2 /* include \0 */, MPI_CHAR, process_num, TAG_SEND_QUERY, MPI_COMM_WORLD); 
            }
            //printf ("%d: Receiving word indexes...\n", my_rank);
            int found = 0; // boolean.
            double word_embeddings [WORD_MATRIX_COL_SIZE];

            #pragma omp for
            for (int process_num = 1; process_num < process_count; ++ process_num) {
                int received_index = -2;
                MPI_Recv (&received_index, 1, MPI_INT, process_num, TAG_SEND_WORD_INDEX, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                // printf ("%d: Received index %d from process %d\n", my_rank, received_index, process_num);
                if (received_index != -1) {
                    //printf ("%d: Found word on process %d\n", my_rank, process_num);
                    MPI_Recv (word_embeddings, WORD_MATRIX_COL_SIZE, MPI_DOUBLE, process_num, TAG_SEND_WORD_EMBEDDINGS, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    //printf ("%d: Received word embeddings, first val is %lf\n", my_rank, word_embeddings [0]);
                    found = 1;
                    // break; // Don't want unread messages.
                }
            }
            if (found) {
                #pragma omp for
                for (int process_num = 1; process_num < process_count; ++ process_num) {
                    // printf ("%d: Sending command to calculate similarity to process %d\n", my_rank, process_num);
                    MPI_Send (&COMMAND_CALCULATE_SIMILARITY, 1, MPI_INT, process_num, TAG_SEND_COMMAND, MPI_COMM_WORLD);
                    /*
                    for (int i = 0; i < WORD_MATRIX_COL_SIZE; ++i) {
                        printf (" %lf ", word_embeddings [i]);
                    }
                    printf ("\n");
                    */
                    //printf ("%d: Sending word embeddings to process %d\n", my_rank, process_num);
                    MPI_Send (word_embeddings, WORD_MATRIX_COL_SIZE, MPI_DOUBLE, process_num, TAG_SEND_WORD_EMBEDDINGS, MPI_COMM_WORLD);
                }
                // Allow them to process in parallel instead of waiting for each
                // just after asking for it.
                #ifdef VERSION_TOP_P
                double top_scores [process_count - 1];
                char top_values   [process_count - 1][MAX_WORD_LENGTH];
                #endif
                #pragma omp for
                for (int process_num = 1; process_num < process_count; ++ process_num) {
                    char most_similar [MAX_WORD_LENGTH] = {0};
                    double score = -1;
                    //printf ("%d: Receiving result from process %d\n", my_rank, process_num);
                    MPI_Recv (most_similar, sizeof (most_similar), MPI_CHAR, process_num, TAG_SEND_WORD, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    //printf ("%d: Receiving score from process %d\n", my_rank, process_num);
                    MPI_Recv (&score, 1, MPI_DOUBLE, process_num, TAG_SEND_SCORE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    printf ("%d: (Top from each process) Received word %s with score %lf from process %d\n", my_rank, most_similar, score, process_num);
                    #ifdef VERSION_TOP_P
                    top_scores [process_num - 1] = score;
                    strcpy (top_values [process_num - 1], most_similar);
                    #endif
                }
                #ifdef VERSION_TOP_P
                for (int score_idx = 0; score_idx < process_count - 1; ++ score_idx) {
                    double max_score = -1000;
                    int process_with_max = 0;
                    for (int process_num = 1; process_num < process_count; ++ process_num) {
                        if (max_score < top_scores [process_num - 1]) {
                            max_score = top_scores [process_num - 1];
                            process_with_max = process_num;

                        }
                    }

                    double * score  = &top_scores [process_with_max - 1];
                    char * most_similar = top_values [process_with_max - 1];
                    printf ("(Top from all) Word %s with score %lf\n", top_values [process_with_max - 1], max_score);
                    for (int i = 0; i < MAX_WORD_LENGTH; ++i) {
                      most_similar [i] = 0;
                    }

                    MPI_Send (&COMMAND_CALCULATE_SIMILARITY, 1, MPI_INT, process_with_max, TAG_SEND_COMMAND, MPI_COMM_WORLD);
                    MPI_Send (word_embeddings, WORD_MATRIX_COL_SIZE, MPI_DOUBLE, process_with_max, TAG_SEND_WORD_EMBEDDINGS, MPI_COMM_WORLD);
                    //printf ("%d: Sent request\n", my_rank);
                    MPI_Recv (most_similar, MAX_WORD_LENGTH, MPI_CHAR, process_with_max, TAG_SEND_WORD, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    //strcpy (top_values [process_with_max], mos)
                    //printf ("%d: Received most similar\n", my_rank);
                    MPI_Recv (score, 1, MPI_DOUBLE, process_with_max, TAG_SEND_SCORE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    //printf ("%d: Received word from process %d\n", my_rank, process_with_max);
                }
                #endif
            } else {
                printf ("Query word was not found\n");
            }
        }
    }
}

int find_word_index (int my_rank, char* words, char* query_word, int range)
{
    for (int index = 0; index < range; ++ index)
    {
        //printf ("%d: %s\n", my_rank, &words[index*MAX_WORD_LENGTH]);
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

    printf ("%d: %s\n", my_rank, "Receiving words...");
    MPI_Recv (words, range * MAX_WORD_LENGTH, MPI_UNSIGNED_CHAR, 0, TAG_SEND_WORDS, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    printf ("%d: %s\n", my_rank, "Receiving matrix...");
    MPI_Recv (word_matrix, range * WORD_MATRIX_COL_SIZE, MPI_DOUBLE, 0, TAG_SEND_MATRIX, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
    /*
    for (int i = 0; i < WORD_MATRIX_COL_SIZE; ++i) {
      printf (" %lf ", word_matrix [i]);
    }
    printf (" > RECV \n");
    */
    #ifdef VERSION_TOP_P
    // Don't resend these positions.
    char sent_already [range];
    #endif
    while(1)
    {
        int command = -1; 
        MPI_Recv (&command, 1, MPI_INT, 0, TAG_SEND_COMMAND, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        //printf ("%d: %s %d\n", my_rank, "Received command", command);
        if (command == COMMAND_EXIT)
            break;
        else if (command == COMMAND_QUERY)
        {
            #ifdef VERSION_TOP_P
            for (int i = 0; i < range; ++i) {
                sent_already [i] = 0;
            }
            #endif
            char query_word [MAX_WORD_LENGTH] = {0};
            MPI_Recv (query_word, MAX_WORD_LENGTH, MPI_CHAR, 0, TAG_SEND_QUERY, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
            //printf ("%d: Received query word %s\n", my_rank, query_word);
            int word_index = find_word_index (my_rank, words, query_word, range);
            //printf ("%d: Sending index %d\n", my_rank, word_index);
            MPI_Send (&word_index, 1, MPI_INT, 0, TAG_SEND_WORD_INDEX, MPI_COMM_WORLD);
            if (word_index != -1) {
                //printf ("%d: Sending word embeddings...\n", my_rank);
                double * to_send = & word_matrix [word_index * WORD_MATRIX_COL_SIZE];
                MPI_Send (to_send, WORD_MATRIX_COL_SIZE, MPI_DOUBLE, 0, TAG_SEND_WORD_EMBEDDINGS, MPI_COMM_WORLD);
            }
        }
        else if (command == COMMAND_CALCULATE_SIMILARITY) {
            double word_embeddings [WORD_MATRIX_COL_SIZE];
            MPI_Recv (word_embeddings, WORD_MATRIX_COL_SIZE, MPI_DOUBLE, 0, TAG_SEND_WORD_EMBEDDINGS, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

           
            int most_similar_index = -1;
            double max_similar_score = -10000;
            for (int row = 0; row < range; ++ row) {
                double similarity = 0;
                for (int col = 0; col < WORD_MATRIX_COL_SIZE; ++ col) {
                    double query_embedding = word_embeddings [col];
                    double emb_matrix      = word_matrix [row * WORD_MATRIX_COL_SIZE + col];
                    //printf ("%d: query_embedding = %lf, emb_matrix = %lf\n", my_rank, query_embedding, emb_matrix);
                    similarity += query_embedding * emb_matrix;
                }
                if (max_similar_score < similarity
                    #ifdef VERSION_TOP_P
                    && sent_already [row] == 0
                    #endif
                ) {
                    max_similar_score = similarity;
                    most_similar_index = row;
                    #ifdef VERSION_TOP_P
                    sent_already [row] = 1;
                    #endif
                }
            }
            //printf ("%d: Most similar index is %d\n", my_rank, most_similar_index);
            char* to_send = &words [most_similar_index * MAX_WORD_LENGTH];
            //printf ("%d: Found word %s as the most similar one with score %lf\n", my_rank, to_send, max_similar_score);
            MPI_Send (to_send, strlen (to_send), MPI_CHAR, 0, TAG_SEND_WORD, MPI_COMM_WORLD);
            MPI_Send (&max_similar_score, 1, MPI_DOUBLE, 0, TAG_SEND_SCORE, MPI_COMM_WORLD);
        } else {
            printf ("%d: Got unknown command: %d\n", my_rank, command);
            exit (3);
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
