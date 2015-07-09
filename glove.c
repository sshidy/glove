/*//  GloVe: Global Vectors for Word Representation
//
//  Copyright (c) 2014 The Board of Trustees of
//  The Leland Stanford Junior University. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//
//
//  For more information, bug reports, fixes, contact:
//    Jeffrey Pennington (jpennin@stanford.edu)
//    GlobalVectors@googlegroups.com
//    http://www-nlp.stanford.edu/projects/glove/ */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#define _FILE_OFFSET_BITS 64
#define MAX_STRING_LENGTH 1000
#define MINLOG 0.0001

typedef double real;

typedef struct cooccur_rec {
    int word1;
    int word2;
    real val;
} CREC;

int verbose = 2; // 0, 1, or 2
int num_threads = 8; // pthreads
int num_iter = 25; // Number of full passes through cooccurrence matrix
int vector_size = 50; // Word vector size
int save_gradsq = 0; // By default don't save squared gradient values
int use_binary = 1; // 0: save as text files; 1: save as binary; 2: both. For binary, save both word and context word vectors.
int model = 2; // For text file output only. 0: concatenate word and context vectors (and biases) i.e. save everything; 1: Just save word vectors (no bias); 2: Save (word + context word) vectors (no biases)
real eta = 0.05; // Initial learning rate
real alpha = 0.75, x_max = 100.0; // Weighting function parameters, not extremely sensitive to corpus, though may need adjustment for very small or very large corpora
real *W, *gradsq, *cost;
long long num_lines, *lines_per_thread, vocab_size;
char *vocab_file, *input_file, *save_W_file, *save_gradsq_file;

/* Efficient string comparison */
int scmp( char *s1, char *s2 ) {
    while(*s1 != '\0' && *s1 == *s2) {s1++; s2++;}
    return(*s1 - *s2);
}

void initialize_parameters() {
	long long a, b;
	vector_size++; // Temporarily increment to allocate space for bias
    
	/* Allocate space for word vectors and context word vectors, and correspodning gradsq 
     * int posix_memalign(void **memptr, size_t alignment, size_t size);
     * posix_memalign() allocates "size" bytes and places the addr of the alloc_mem in *memptr. 
     * The addr will be a multiple of "alignment", which must be 2^k and a multiple of sizeof(void *). 
     * If size=0, it returns either NULL or a unique pointer that can later be successfully passed to free(3).
     * returns zero on success, or 
     * EINVAL: "alignment" was not 2^k, or was not a multiple of sizeof(void *). ENOMEM: insufficient memory
     */
	a = posix_memalign((void **)&W, 128, 2 * vocab_size * vector_size * sizeof(real)); // Might perform better than malloc
    if (W == NULL) {
        fprintf(stderr, "Error allocating memory for W\n");
        exit(1);
    }
    a = posix_memalign((void **)&gradsq, 128, 2 * vocab_size * vector_size * sizeof(real)); // Might perform better than malloc
	if (gradsq == NULL) {
        fprintf(stderr, "Error allocating memory for gradsq\n");
        exit(1);
    }
	for (b = 0; b < vector_size; b++) for (a = 0; a < 2 * vocab_size; a++) W[a * vector_size + b] = (rand()/(real)RAND_MAX - 0.5) / vector_size;//-0.5~0.5/vector_size
	for (b = 0; b < vector_size; b++) for (a = 0; a < 2 * vocab_size; a++) gradsq[a * vector_size + b] = 1.0; // So init eta = init learn_rate //AdaGrad
	vector_size--; //
}

/* Train the GloVe model */
void *glove_thread(void *vid) { short fg=0;
    long long a, b ,l1, l2;
    long long id = (long long) vid;
    CREC cr;
    real diff, fdiff, temp1, temp2, tp;
    FILE *fin;
    fin = fopen(input_file, "rb");
    fseeko(fin, (num_lines / num_threads * id) * (sizeof(CREC)), SEEK_SET); //Threads spaced roughly equally throughout file
    cost[id] = 0;
    
    for(a = 0; a < lines_per_thread[id]; a++) { //
        fread(&cr, sizeof(CREC), 1, fin);
        if(feof(fin)) break;
        if (cr.val<MINLOG) fg=1; //abnormal!
        if ((cr.word1<=0)||(cr.word2<=0)) fprintf(stderr, "word1=%d,word2=%d,val=%lf\n",cr.word1,cr.word2,cr.val); //should not exist!
        /* Get location of words in W & gradsq */
        l1 = (cr.word1 - 1LL) * (vector_size + 1); // cr word indices start at 1 //1LL: long long int 1, force word1 to be LL
        l2 = ((cr.word2 - 1LL) + vocab_size) * (vector_size + 1); // shift by vocab_size to get separate vectors for context words //half way of W[]
        if (fg==1) fprintf(stderr,"l1=%lld,l2=%lld\n",l1,l2); //find abnormal position
        /* Calculate cost, save diff for gradients */
        diff = 0;
        for(b = 0; b < vector_size; b++) {diff += W[b + l1] * W[b + l2];} // dot product of word and context word vector //w_i^T * w_j
        if (fg!=1){
            diff += W[vector_size + l1] + W[vector_size + l2] - log(cr.val); // add separate bias for each word //(8)
            fdiff = (cr.val > x_max) ? diff : pow(cr.val / x_max, alpha) * diff; // multiply weighting function (f) with diff //(9) f(X_ij)*diff
            cost[id] += 0.5 * fdiff * diff; // weighted squared error //f*diff*diff an ultra 0.5 here
        }else { //avoid log(too_small), and show the cooccur data!
            fdiff = -cr.val*7; 
            fprintf(stderr,"%lld: word1=%d,word2=%d,val=%lf, W1=%lf,W2=%lf\n",a,cr.word1,cr.word2,cr.val,W[vector_size+l1],W[vector_size+l2]);
        }
        
        /* Adaptive gradient updates */ //AdaGrade
        fdiff *= eta; // for ease in calculating gradient //eta*f*diff *\par{diff}/\par{W1, W2, b1, b2}
        for(b = 0; b < vector_size; b++) {
            // learning rate times gradient for word vectors
            temp1 = fdiff * W[b + l2];
            temp2 = fdiff * W[b + l1];
            // adaptive updates //AdaGrade: /sqrt(sum {history grad}^2)
            tp=sqrt(gradsq[b + l1]);
            if (tp>=1) W[b + l1] -= temp1/tp; 
            else {W[b + l1] -= temp1; fprintf(stderr,"%lld: pos:%lld,W1_gradsq=%lf\n",a,b+l1,gradsq[b+l1]);} 
            tp=sqrt(gradsq[b + l2]);
            if (tp>=1) W[b + l2] -= temp2/tp; 
            else {W[b + l2] -= temp2; fprintf(stderr,"%lld: pos:%lld,W2_gradsq=%lf\n",a,b+l2,gradsq[b+l2]);}
            gradsq[b + l1] += temp1 * temp1;
            gradsq[b + l2] += temp2 * temp2;
        }  
        if (fg==1) fprintf(stderr,"l1=%lld,l2=%lld\n",l1,l2);
        // updates for bias terms
        tp=sqrt(gradsq[vector_size + l1]);
        if (tp>=1) W[vector_size + l1] -= fdiff/tp; 
        else {W[vector_size + l1] -= fdiff; fprintf(stderr,"%lld: pos:%lld,bios1_gradsq=%lf\n",a,vector_size+l1,gradsq[vector_size+l1]);}
        tp=sqrt(gradsq[vector_size + l2]);
        if (tp>=1) W[vector_size + l2] -= fdiff/tp; 
        else {W[vector_size + l2] -= fdiff; fprintf(stderr,"%lld: pos:%lld,bios2_gradsq=%lf\n",a,vector_size+l2,gradsq[vector_size+l2]);}
        fdiff *= fdiff;
        gradsq[vector_size + l1] += fdiff;
        gradsq[vector_size + l2] += fdiff;
        fg=0; //end of abnormal
    }
    
    fclose(fin);
    pthread_exit(NULL);
}

/* Save params to file */
int save_params() {
    long long a, b;
    char format[20];
    char output_file[MAX_STRING_LENGTH], output_file_gsq[MAX_STRING_LENGTH];
    char *word = malloc(sizeof(char) * MAX_STRING_LENGTH);
    FILE *fid, *fout, *fgs;
    
    if(use_binary > 0) { // Save parameters in binary file
        sprintf(output_file,"%s.bin",save_W_file); //save_W_file: vectors
        fout = fopen(output_file,"wb");
        if(fout == NULL) {fprintf(stderr, "Unable to open file %s.\n",save_W_file); return 1;}
        for(a = 0; a < 2 * (long long)vocab_size * (vector_size + 1); a++) fwrite(&W[a], sizeof(real), 1,fout); //write W[0] 1-by-1, then W[1],...
        fclose(fout);
        if(save_gradsq > 0) {
            sprintf(output_file_gsq,"%s.bin",save_gradsq_file);
            fgs = fopen(output_file_gsq,"wb");
            if(fgs == NULL) {fprintf(stderr, "Unable to open file %s.\n",save_gradsq_file); return 1;}
            for(a = 0; a < 2 * (long long)vocab_size * (vector_size + 1); a++) fwrite(&gradsq[a], sizeof(real), 1,fgs);
            fclose(fgs);
        }
    }
    if(use_binary != 1) { // Save parameters in text file
        sprintf(output_file,"%s.txt",save_W_file);
        if(save_gradsq > 0) {
            sprintf(output_file_gsq,"%s.txt",save_gradsq_file);
            fgs = fopen(output_file_gsq,"wb");
            if(fgs == NULL) {fprintf(stderr, "Unable to open file %s.\n",save_gradsq_file); return 1;}
        }
        fout = fopen(output_file,"wb");
        if(fout == NULL) {fprintf(stderr, "Unable to open file %s.\n",save_W_file); return 1;}
        fid = fopen(vocab_file, "r");
        sprintf(format,"%%%ds",MAX_STRING_LENGTH);
        if(fid == NULL) {fprintf(stderr, "Unable to open file %s.\n",vocab_file); return 1;}
        for(a = 0; a < vocab_size; a++) {
            if(fscanf(fid,format,word) == 0) return 1;
            fprintf(fout, "%s",word);
            if(model == 0) { // Save all parameters (including bias)
                for(b = 0; b < (vector_size + 1); b++) fprintf(fout," %lf", W[a * (vector_size + 1) + b]); //the last one is bias
                for(b = 0; b < (vector_size + 1); b++) fprintf(fout," %lf", W[(vocab_size + a) * (vector_size + 1) + b]);
            }
            if(model == 1) // Save only "word" vectors (without bias)
                for(b = 0; b < vector_size; b++) fprintf(fout," %lf", W[a * (vector_size + 1) + b]);
            if(model == 2) // Save "word + context word" vectors (without bias)
                for(b = 0; b < vector_size; b++) fprintf(fout," %lf", W[a * (vector_size + 1) + b] + W[(vocab_size + a) * (vector_size + 1) + b]);
            fprintf(fout,"\n");
            if(save_gradsq > 0) { // Save gradsq
                fprintf(fgs, "%s",word);
                for(b = 0; b < (vector_size + 1); b++) fprintf(fgs," %lf", gradsq[a * (vector_size + 1) + b]);
                for(b = 0; b < (vector_size + 1); b++) fprintf(fgs," %lf", gradsq[(vocab_size + a) * (vector_size + 1) + b]);
                fprintf(fgs,"\n");
            }
            if(fscanf(fid,format,word) == 0) return 1; // Eat irrelevant frequency entry
        }
        fclose(fid);
        fclose(fout);
        if(save_gradsq > 0) fclose(fgs);
    }
    return 0;
}

/* Train model */
int train_glove() {
    long long a, file_size;
    int b;
    FILE *fin;
    real total_cost = 0;
    fprintf(stderr, "TRAINING MODEL\n");
    
    fin = fopen(input_file, "rb");
    if(fin == NULL) {fprintf(stderr,"Unable to open cooccurrence file %s.\n",input_file); return 1;}
    /* fseeko() & ftello() are identical to fseek(3) and ftell(3) respectively, except that the offset arg of fseeko() and the return val of ftello() is of type off_t instead of long.
       On some archs, both off_t and long are 32-bit, but defining _FILE_OFFSET_BITS=64 (before including any header files) will turn off_t into 64-bit.    */
    fseeko(fin, 0, SEEK_END);
    file_size = ftello(fin); //the 2 statems are std to get file_size
    num_lines = file_size/(sizeof(CREC)); // Assuming the file isn't corrupt and consists only of CREC's
    fclose(fin);
    fprintf(stderr,"Read %lld lines.\n", num_lines);
    if(verbose > 1) fprintf(stderr,"Initializing parameters...");
    initialize_parameters();
    if(verbose > 1) fprintf(stderr,"done.\n");
    if(verbose > 0) fprintf(stderr,"vector size: %d\n", vector_size);
    if(verbose > 0) fprintf(stderr,"vocab size: %lld\n", vocab_size);
    if(verbose > 0) fprintf(stderr,"x_max: %lf\n", x_max);
    if(verbose > 0) fprintf(stderr,"alpha: %lf\n", alpha);
    pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t)); //8*sizeof(pthread_t)
    lines_per_thread = (long long *) malloc(num_threads * sizeof(long long)); //each stores the line_num that a thread will proc
    
    // Lock-free asynchronous SGD
    for(b = 0; b < num_iter; b++) { // 15 iterations
        total_cost = 0;
        for (a = 0; a < num_threads - 1; a++) lines_per_thread[a] = num_lines / num_threads; // 60666466/8
        lines_per_thread[a] = num_lines / num_threads + num_lines % num_threads; //last one is biggest
        for (a = 0; a < num_threads; a++) {pthread_create(&pt[a], NULL, glove_thread, (void *)a); } //fprintf(stderr,"thread %lld created\n",a);
        for (a = 0; a < num_threads; a++) { pthread_join(pt[a], NULL); } //fprintf(stderr,"thread %lld is to be joined\n",a);
        for (a = 0; a < num_threads; a++) {total_cost += cost[a]; } // fprintf(stderr,"cost %lld: %lf\n", a, cost[a]);
        fprintf(stderr,"iter: %03d, cost: %lf\n", b+1, total_cost/num_lines);
    }
    return save_params();
}

int find_arg(char *str, int argc, char **argv) {
    int i;
    for (i = 1; i < argc; i++) {
        if(!scmp(str, argv[i])) {
            if (i == argc - 1) {
                printf("No argument given for %s\n", str);
                exit(1);
            }
            return i;
        }
    }
    return -1;
}

int main(int argc, char **argv) {
    int i;
    FILE *fid;
    vocab_file = malloc(sizeof(char) * MAX_STRING_LENGTH);
    input_file = malloc(sizeof(char) * MAX_STRING_LENGTH);
    save_W_file = malloc(sizeof(char) * MAX_STRING_LENGTH);
    save_gradsq_file = malloc(sizeof(char) * MAX_STRING_LENGTH);
    
    if (argc == 1) {
        printf("GloVe: Global Vectors for Word Representation, v0.2\n");
        printf("Author: Jeffrey Pennington (jpennin@stanford.edu)\n\n");
        printf("Usage options:\n");
        printf("\t-verbose <int>\n");
        printf("\t\tSet verbosity: 0, 1, or 2 (default)\n");
        printf("\t-vector-size <int>\n");
        printf("\t\tDimension of word vector representations (excluding bias term); default 50\n");
        printf("\t-threads <int>\n");
        printf("\t\tNumber of threads; default 8\n");
        printf("\t-iter <int>\n");
        printf("\t\tNumber of training iterations; default 25\n");
        printf("\t-eta <float>\n");
        printf("\t\tInitial learning rate; default 0.05\n");
        printf("\t-alpha <float>\n");
        printf("\t\tParameter in exponent of weighting function; default 0.75\n");
        printf("\t-x-max <float>\n");
        printf("\t\tParameter specifying cutoff in weighting function; default 100.0\n");
        printf("\t-binary <int>\n");
        printf("\t\tSave output in binary format (0: text, 1: binary, 2: both); default 0\n");
        printf("\t-model <int>\n");
        printf("\t\tModel for word vector output (for text output only); default 2\n");
        printf("\t\t   0: output all data, for both word and context word vectors, including bias terms\n");
        printf("\t\t   1: output word vectors, excluding bias terms\n");
        printf("\t\t   2: output word vectors + context word vectors, excluding bias terms\n");
        printf("\t-input-file <file>\n");
        printf("\t\tBinary input file of shuffled cooccurrence data (produced by 'cooccur' and 'shuffle'); default cooccurrence.shuf.bin\n");
        printf("\t-vocab-file <file>\n");
        printf("\t\tFile containing vocabulary (truncated unigram counts, produced by 'vocab_count'); default vocab.txt\n");
        printf("\t-save-file <file>\n");
        printf("\t\tFilename, excluding extension, for word vector output; default vectors\n");
        printf("\t-gradsq-file <file>\n");
        printf("\t\tFilename, excluding extension, for squared gradient output; default gradsq\n");
        printf("\t-save-gradsq <int>\n");
        printf("\t\tSave accumulated squared gradients; default 0 (off); ignored if gradsq-file is specified\n");
        printf("\nExample usage:\n");
        printf("./glove -input-file cooccurrence.shuf.bin -vocab-file vocab.txt -save-file vectors -gradsq-file gradsq -verbose 2 -vector-size 100 -threads 16 -alpha 0.75 -x-max 100.0 -eta 0.05 -binary 2 -model 2\n\n");
        return 0;
    }
    
    
    if ((i = find_arg((char *)"-verbose", argc, argv)) > 0) verbose = atoi(argv[i + 1]);
    if ((i = find_arg((char *)"-vector-size", argc, argv)) > 0) vector_size = atoi(argv[i + 1]);
    if ((i = find_arg((char *)"-iter", argc, argv)) > 0) num_iter = atoi(argv[i + 1]);
    if ((i = find_arg((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
    cost = malloc(sizeof(real) * num_threads);
    if ((i = find_arg((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
    if ((i = find_arg((char *)"-x-max", argc, argv)) > 0) x_max = atof(argv[i + 1]); //"we fix to x_max = 100 for all our experiments"
    if ((i = find_arg((char *)"-eta", argc, argv)) > 0) eta = atof(argv[i + 1]);
    if ((i = find_arg((char *)"-binary", argc, argv)) > 0) use_binary = atoi(argv[i + 1]);
    if ((i = find_arg((char *)"-model", argc, argv)) > 0) model = atoi(argv[i + 1]);
    if(model != 0 && model != 1) model = 2;
    if ((i = find_arg((char *)"-save-gradsq", argc, argv)) > 0) save_gradsq = atoi(argv[i + 1]);
    if ((i = find_arg((char *)"-vocab-file", argc, argv)) > 0) strcpy(vocab_file, argv[i + 1]);
    else strcpy(vocab_file, (char *)"vocab.txt");
    if ((i = find_arg((char *)"-save-file", argc, argv)) > 0) strcpy(save_W_file, argv[i + 1]);
    else strcpy(save_W_file, (char *)"vectors");
    if ((i = find_arg((char *)"-gradsq-file", argc, argv)) > 0) {
        strcpy(save_gradsq_file, argv[i + 1]);
        save_gradsq = 1;
    }
    else if(save_gradsq > 0) strcpy(save_gradsq_file, (char *)"gradsq");
    if ((i = find_arg((char *)"-input-file", argc, argv)) > 0) strcpy(input_file, argv[i + 1]);
    else strcpy(input_file, (char *)"cooc.shuf.bin");
    
    vocab_size = 0;
    fid = fopen(vocab_file, "r");
    if(fid == NULL) {fprintf(stderr, "Unable to open vocab file %s.\n",vocab_file); return 1;}
    while ((i = getc(fid)) != EOF) if (i == '\n') vocab_size++; // Count number of entries in vocab_file
    fclose(fid);
    
    return train_glove();
}
