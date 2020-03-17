#include <mpi.h>
#include <stdio.h>


#include <stdlib.h>
#include <string.h>
#include <errno.h>


const int MAX_LINE_LEN = 6000;
const int MAX_WORD_LEN = 20;
const int NUM_WORDS = 10;
const int EMBEDDING_DIMENSION= 300;
const char DELIMITER[2] = "\t";

const int COMMAND_QUERY= 1;


void distributeEmbeddings(char *filename){
	char line[MAX_LINE_LEN];
	printf( "Reading embedding file\n");
	
	FILE *file = fopen (filename, "r" );
	int wordIndex = 0;
    int p = 1;
    printf("Sending the necessary data to process %d\n",p);
    float* embeddings_matrix = (float*)malloc(sizeof(float) * NUM_WORDS*EMBEDDING_DIMENSION);

    char* words = (char*)malloc(sizeof(char) * NUM_WORDS*MAX_WORD_LEN);
    
    for(int i = 0; i<NUM_WORDS; i++){
      fgets(line, MAX_LINE_LEN, file);

      char *word;
      word = strtok(line, DELIMITER);
      strcpy(words+i*MAX_WORD_LEN, word);
      for(int embIndex = 0; embIndex<EMBEDDING_DIMENSION; embIndex++){
	       char *field = strtok(NULL, DELIMITER);
	       float emb = strtof(field,NULL);
	       *(embeddings_matrix+i*EMBEDDING_DIMENSION+embIndex) = emb;
      }
	}
    
    printf("Sending words to process... %d\n",1);
    MPI_Send(
         	/* data         = */ words, 
      		/* count        = */ NUM_WORDS*MAX_WORD_LEN, 
      		/* datatype     = */ MPI_CHAR,
      		/* destination  = */ p, 
      		/* tag          = */ 0, 
      		/* communicator = */ MPI_COMM_WORLD);

    printf("Sending embeddings to process %d\n",p);

    MPI_Send(
         	/* data         = */ embeddings_matrix, 
      		/* count        = */ NUM_WORDS*EMBEDDING_DIMENSION, 
      		/* datatype     = */ MPI_FLOAT,
      		/* destination  = */ p,
      		/* tag          = */ 0,
      		/* communicator = */ MPI_COMM_WORLD);
   
   printf( "Embedding file.. has been distributed\n");
}




 int findWordIndex(char *words, char *query_word){
   for(int wordIndex = 0; wordIndex<NUM_WORDS; wordIndex++){
    if(strcmp((words+wordIndex*MAX_WORD_LEN), query_word)==0){
     return wordIndex;
   }
 }
 return -1;
}


void runMasterNode(int world_rank){

    // If we are rank 0, set the number to -1 and send it to process 1

    distributeEmbeddings("./word_embeddings_small.txt");

    while(1==1){
            printf("Please type a query word:\n");
            char queryWord[256];
            scanf( "%s" , queryWord);
            printf("Query word:%s\n",queryWord);

            int p = 1;
	        printf("Command %d is being sent to process %d:\n", COMMAND_QUERY, p);
	        MPI_Send(
	                  /* data         = */ (void *)&COMMAND_QUERY, 
	                  /* count        = */ 1, 
	                  /* datatype     = */ MPI_INT, 
	                  /* destination  = */ p,
	                  /* tag          = */ 0, 
	                  /* communicator = */ MPI_COMM_WORLD);

	        printf("Command %d sent to process %d:\n", COMMAND_QUERY, p);
	              
	        MPI_Send(
	                  /* data         = */ queryWord, 
	                  /* count        = */ MAX_WORD_LEN, 
	                  /* datatype     = */ MPI_CHAR, 
	                  /* destination  = */ p,
	                  /* tag          = */ 0, 
	                  /* communicator = */ MPI_COMM_WORLD);

	        printf("Query %s sent to process %d:\n", queryWord, p);


	        char* words = (char*)malloc(sizeof(char) * NUM_WORDS*MAX_WORD_LEN);

	        
	        MPI_Recv(
	                  /* data         = */ words, 
	                  /* count        = */ NUM_WORDS*MAX_WORD_LEN, 
	                  /* datatype     = */ MPI_CHAR, 
	                  /* source       = */ p, 
	                  /* tag          = */ 0, 
	                  /* communicator = */ MPI_COMM_WORLD, 
	                  /* status       = */ MPI_STATUS_IGNORE);

	        float* similarityScores = (float*)malloc(sizeof(float) * NUM_WORDS);


	        MPI_Recv(
	                  /* data         = */ similarityScores, 
	                  /* count        = */ NUM_WORDS, 
	                  /* datatype     = */ MPI_FLOAT, 
	                  /* source       = */ p, 
	                  /* tag          = */ 0, 
	                  /* communicator = */ MPI_COMM_WORLD, 
	                  /* status       = */ MPI_STATUS_IGNORE);

	        


	        printf("Query results: ===============\n");
	        

	        for(int i = 0; i<NUM_WORDS; i++){
	        	printf("****** word: %s, similarity: %f\n", (words+i*MAX_WORD_LEN), *(similarityScores+i));
	        }
	        
	        printf("=============================\n");
	        free(similarityScores);
	        free(words);
   }
}




void runSlaveNode(int world_rank){


  printf("Receiving words in process #: %d\n", world_rank);

  char* words = (char*)malloc(sizeof(char) * NUM_WORDS*MAX_WORD_LEN);
  
  MPI_Recv(words, NUM_WORDS*MAX_WORD_LEN, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  printf("Words received in process #: %d\n", world_rank);

  float* embeddings_matrix = (float*)malloc(sizeof(float) * NUM_WORDS*EMBEDDING_DIMENSION);

  printf("Process %d started to receive embedding part\n", world_rank);
  
  // Now receive the message with the allocated buffer
  MPI_Recv(embeddings_matrix, NUM_WORDS*EMBEDDING_DIMENSION, MPI_FLOAT, 0, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);

  printf("Process %d received embedding part\n", world_rank);


  while(1==1){

      printf("Process %d is waiting for command\n", world_rank);

      int command;

      MPI_Recv(
          /* data         = */ &command, 
          /* count        = */ 1, 
          /* datatype     = */ MPI_INT, 
          /* source       = */ 0, 
          /* tag          = */ 0, 
          /* communicator = */ MPI_COMM_WORLD, 
          /* status       = */ MPI_STATUS_IGNORE);
      printf("Command received:%d\n",command);

      if(command == COMMAND_QUERY)
      {

      	  char* query_word = (char*)malloc(sizeof(char) * MAX_WORD_LEN);

          MPI_Recv(
          /* data         = */ query_word, 
          /* count        = */ MAX_WORD_LEN, 
          /* datatype     = */ MPI_CHAR, 
          /* source       = */ 0, 
          /* tag          = */ 0, 
          /* communicator = */ MPI_COMM_WORLD, 
          /* status       = */ MPI_STATUS_IGNORE);

          printf("Query word received:%s\n",query_word);

          int wordIndex = findWordIndex(words, query_word);

          float* query_embeddings = (embeddings_matrix+wordIndex*EMBEDDING_DIMENSION);
          

          printf("Query embeddings was found\n");

          printf("Calculating similarities in process :%d\n", world_rank);
          int mostSimilarWordIndex = -1;
          float maxSimilarityScore = 0;

          float* similarityScores = (float*)malloc(sizeof(float) * NUM_WORDS);

          for(int wordIndex = 0; wordIndex<NUM_WORDS; wordIndex++){
            float similarity = 0.0;
            for(int embIndex = 0; embIndex<EMBEDDING_DIMENSION; embIndex++){
             float emb1 = *(query_embeddings + embIndex);
             float emb2 = *(embeddings_matrix + wordIndex*EMBEDDING_DIMENSION + embIndex);
             similarity +=(emb1*emb2);
            }

            *(similarityScores + wordIndex) = similarity;
         }

         printf("similarities was calculated in process :%d\n", world_rank);


         printf("words are being sent back to master node in process:%d\n", world_rank);
         MPI_Send(
         	/* data         = */ words, 
      		/* count        = */ NUM_WORDS*MAX_WORD_LEN, 
      		/* datatype     = */ MPI_CHAR,
      		/* destination  = */ 0, 
      		/* tag          = */ 0, 
      		/* communicator = */ MPI_COMM_WORLD);

         //printf("similarity scores are sent to master node in process:%d\n", world_rank);
      	 MPI_Send(
         	/* data         = */ similarityScores, 
      		/* count        = */ NUM_WORDS, 
      		/* datatype     = */ MPI_FLOAT,
      		/* destination  = */ 0, 
      		/* tag          = */ 0, 
      		/* communicator = */ MPI_COMM_WORLD);
      }

  }

  free(embeddings_matrix);

}

int main(int argc, char** argv) {

    // Initialize the MPI environment
  MPI_Init(NULL, NULL);

    // Get the number of processes
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Get the name of the processor
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);

    // Print off a hello world message
  printf("Hello world from processor %s, rank %d out of %d processors\n",
   processor_name, world_rank, world_size);


    // We are assuming at least 2 processes for this task
  if (world_size < 2) {
    fprintf(stderr, "World size must be greater than 1 for %s\n", argv[0]);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  int wordIndex;

  if (world_rank == 0) {
    runMasterNode(world_rank);


  } else {

    runSlaveNode(world_rank);

  }
      
    
    MPI_Finalize();



    printf("Process %d stopped\n", world_rank);
  }