/*
 * ex2.c
 *
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <curl/curl.h>

#define HTTP_OK 200L
#define REQUEST_TIMEOUT_SECONDS 2L

#define URL_OK 0
#define URL_ERROR 1
#define URL_UNKNOWN 2

#define MAX_PROCESSES 1024

typedef struct {
	int ok, error, unknown;
} UrlStatus;

void usage() {
	fprintf(stderr, "usage:\n\t./ex2 FILENAME NUMBER_OF_PROCESSES\n");
	exit(EXIT_FAILURE);
}

int check_url(const char *url) {
	CURL *curl;
	CURLcode res;
	long response_code = 0L;

	curl = curl_easy_init();

	if(curl) {
		curl_easy_setopt(curl, CURLOPT_URL, url);
		curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
		curl_easy_setopt(curl, CURLOPT_TIMEOUT, REQUEST_TIMEOUT_SECONDS);
		curl_easy_setopt(curl, CURLOPT_NOBODY, 1L); /* do a HEAD request */

		res = curl_easy_perform(curl);
		if(res == CURLE_OK) {
			curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);
			if (response_code == HTTP_OK) {
				return URL_OK;
			} else {
				return URL_ERROR;
			}
		}

		curl_easy_cleanup(curl);
	}

	return URL_UNKNOWN;
}

void serial_checker(const char *filename) {
	UrlStatus results = {0};
	FILE *toplist_file;
	char *line = NULL;
	size_t len = 0;
	ssize_t read;

	toplist_file = fopen(filename, "r");

	if (toplist_file == NULL) {
		exit(EXIT_FAILURE);
	}

	while ((read = getline(&line, &len, toplist_file)) != -1) {
		if (read == -1) {
			perror("unable to read line from file");
		}
		line[read-1] = '\0'; /* null-terminate the URL */
		switch (check_url(line)) {
		case URL_OK:
			results.ok += 1;
			break;
		case URL_ERROR:
			results.error += 1;
			break;
		default:
			results.unknown += 1;
		}
	}

	free(line);
	fclose(toplist_file);

	printf("%d OK, %d Error, %d Unknown\n",
			results.ok,
			results.error,
			results.unknown);
}

void worker_checker(const char *filename, int pipe_write_fd, int worker_id, int workers_number) {

	 UrlStatus results = {0};
	 FILE *toplist_file;
	 char *line = NULL;
   size_t len = 0;
	 ssize_t read;
   toplist_file = fopen(filename, "r");

 	if (toplist_file == NULL) {
 		 perror("Error occured opening the file");
 		 exit(EXIT_FAILURE);
 	}

	// Counts the number of line in the file
 	int lines = 0;
 	while ((read = getline(&line, &len, toplist_file)) != -1) {
 		lines++;
 	}


 	fseek(toplist_file, 0, SEEK_SET);
	int lines_to_read = (int) (lines / workers_number);
	int startIndex = lines_to_read * worker_id;

	for (int i = 0; i < startIndex; i++) {
		if(getline(&line, &len, toplist_file) == -1){
			perror("Error occured while reading the file");
			exit(EXIT_FAILURE);
		}
	}

	lines_to_read +=  worker_id == workers_number - 1 ? lines % workers_number : 0;

 	for (int j = 0; j < lines_to_read; j++) {
		read = getline(&line, &len, toplist_file);
		if (read == -1) {
			perror("Error occured while reading the file");
			exit(EXIT_FAILURE);
		}
		line[read-1] = '\0'; /* null-terminate the URL */

 	 switch (check_url(line)) {
 	 case URL_OK:
 		 results.ok += 1;
 		 break;
 	 case URL_ERROR:
 		 results.error += 1;
 		 break;
 	 default:
 		 results.unknown += 1;
 	 }
 	}

	// Close the file
 	free(line);
 	fclose(toplist_file);
	// Write back the results to the pipe
 	write(pipe_write_fd, &results, sizeof(results));
}


void parallel_checker(const char *filename, int number_of_processes) {

	 // Creates the pipe
	 int fileDescriptor[2];
	 pid_t pids[number_of_processes];
	 if (pipe(fileDescriptor) == -1) {
		 perror("Failure in creating the pipe");
		 exit(EXIT_FAILURE);
	 }

	 // Creates number_of_processes child processes
   for (int i = 0; i < number_of_processes; i++) {
		 pids[i] = fork();
		 if (pids[i] < 0) { // Checks for error
			 perror("Error occured while forking");
			 exit(EXIT_FAILURE);
		 }else if (pids[i] == 0) { // Only child executes this part
			 close(fileDescriptor[0]);
			 worker_checker(filename, fileDescriptor[1], i, number_of_processes);
			 close(fileDescriptor[1]);
			 exit(EXIT_SUCCESS);
		 }
	 }

	 // Only parent process executes this part
	 int stat = 0;
	 UrlStatus results = {0};
	 UrlStatus tempStruct = {0};
	 // Wait for each process to terminate and write back the results to the struct
	 for (int i = 0; i < number_of_processes; i++) {
			wait(NULL);
			stat = read(fileDescriptor[0], &tempStruct, sizeof(tempStruct));
			if (stat == -1) {
				perror("Error occured reading the pipe");
 			  exit(EXIT_FAILURE);
			}
			results.ok += tempStruct.ok;
	 	  results.error += tempStruct.error;
	 	  results.unknown += tempStruct.unknown;
	 }
	 close(fileDescriptor[1]);
	 close(fileDescriptor[0]);
	 printf("%d OK, %d Error, %d Unknown\n", results.ok, results.error, results.unknown);
   exit(EXIT_SUCCESS);
}

int main(int argc, char **argv) {
	if (argc != 3) {
		usage();
	} else if (atoi(argv[2]) == 1) {
		serial_checker(argv[1]);
	} else {
		parallel_checker(argv[1], atoi(argv[2]));
	}

	return EXIT_SUCCESS;
}
