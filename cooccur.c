/*//  Tool to calculate word-word cooccurrence statistics
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
//    http://www-nlp.stanford.edu/projects/glove/   */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define TSIZE 1048576
#define SEED 1159241
#define HASHFN bitwisehash
#define INTERV 5000000
#define MAX_STRING_LENGTH 32

//static const int MAX_STRING_LENGTH = 1000; //should be large enough, and cannot use it to init an array
typedef double real;

typedef struct cooccur_rec {
    int word1;  //rank num in the vocab
    int word2;
    real val;  // sum(1/distance(word1,word2))
} CREC;

typedef struct cooccur_rec_id {
    int word1;
    int word2;
    real val;
    int id;  //ordinal num of temp (overflow) file
} CRECID;

typedef struct hashrec {
    char word[MAX_STRING_LENGTH];    //char *word
    long long id;
    struct hashrec *next;
} HASHREC;

int verbose = 2, cmp_times=1, twoexp=1; // 0, 1, or 2
long long max_product; // Cutoff for product of word frequency ranks below which cooccurrence counts will be stored in a compressed full array
long long overflow_length; // Number of cooccurrence records whose product exceeds max_product to store in memory before writing to disk
int window_size = 15; // default context window size
int symmetric = 1; // 0: asymmetric, 1: symmetric
real memory_limit = 3; // soft limit, in gigabytes, used to estimate optimal array sizes
char *vocab_file, *file_head;

/* Efficient string comparison */
int scmp( char *s1, char *s2 ) {
    while(*s1 != '\0' && *s1 == *s2) {s1++; s2++;}
    return(*s1 - *s2);
}

/* Move-to-front hashing and hash function from Hugh Williams, http://www.seg.rmit.edu.au/code/zwh-ipl/ */

/* Simple bitwise hash function */
unsigned int bitwisehash(char *word, int tsize, unsigned int seed) {
    char c;
    unsigned int h;
    h = seed;
    for(; (c =* word) != '\0'; word++) h ^= ((h << 5) + c + (h >> 2));
    return((unsigned int)((h&0x7fffffff) % tsize));
}

/* Create hash table, initialise pointers to NULL */
HASHREC ** inithashtable() {
    int	i;
    HASHREC **ht;
    ht = (HASHREC **) malloc( sizeof(HASHREC *) * TSIZE ); //vocab_hash, freed
    for(i = 0; i < TSIZE; i++) ht[i] = (HASHREC *) NULL;
    return(ht);
}

/* Search hash table for given string, return record if found, else NULL */
HASHREC *hashsearch(HASHREC **ht, char *w) {
    HASHREC	*htmp, *hprv;
    unsigned int hval = HASHFN(w, TSIZE, SEED); if (hval>TSIZE-1) {fprintf(stderr,"unsigned wrong!\n"); exit(1);}
    for(hprv = NULL, htmp=ht[hval]; htmp != NULL && scmp(htmp->word, w) != 0; hprv = htmp, htmp = htmp->next);
    if( htmp != NULL && hprv!=NULL ) { // move to front on access
        hprv->next = htmp->next;
        htmp->next = ht[hval];
        ht[hval] = htmp;
    }
    return(htmp);
}

/* Insert string in hash table, check for duplicates which should be absent */
void hashinsert(HASHREC **ht, char *w, long long id) {
    HASHREC	*htmp, *hprv;
    unsigned int hval = HASHFN(w, TSIZE, SEED); if (hval>TSIZE-1) {fprintf(stderr,"Unsigned wrong!\n"); exit(1);}
    for(hprv = NULL, htmp = ht[hval]; htmp != NULL && scmp(htmp->word, w) != 0; hprv = htmp, htmp = htmp->next);
    if(htmp == NULL) {
        htmp = (HASHREC *) malloc(sizeof(HASHREC));
        //htmp->word = (char *) malloc(strlen(w) + 1);
        if (strlen(w)>MAX_STRING_LENGTH-1) {fprintf(stderr, "The word %s is too long, change MAX_STRING_LENGTH!\n",w); exit(1);}
        strcpy(htmp->word, w);
        htmp->id = id;
        htmp->next = NULL;
        if(hprv == NULL) ht[hval] = htmp;
        else hprv->next = htmp;
    }
    else fprintf(stderr, "Error, duplicate entry located: %s.\n",htmp->word);
    return;
}

/* Read word from input stream */
int get_word(char *word, FILE *fin) {
    int i = 0, ch;
    while(!feof(fin)) {
        ch = fgetc(fin);
        if(ch == 13) continue; //"carriage return"
        if((ch == ' ') || (ch == '\t') || (ch == '\n')) {
            if(i > 0) {
                if (ch == '\n') ungetc(ch, fin);
                break;
            }
            if (ch == '\n') return 1;  //return|newline, flag=1
            else continue;
        }
        word[i++] = ch;
        if(i >= MAX_STRING_LENGTH - 1) i--;  // truncate words that exceed max length //long "words" in raw text, usu. discard them
    }
    word[i] = 0; //add '\0' at end
    return 0;
}

/* Write sorted chunk of cooccurrence records to file, accumulating duplicate entries */
int write_chunk(CREC *cr, long long length, FILE *fout) {
    long long a = 0;
    CREC old = cr[a];
    
    for(a = 1; a < length; a++) {
        if(cr[a].word1 == old.word1 && cr[a].word2 == old.word2) {
            old.val += cr[a].val;
            continue;
        }
        fwrite(&old, sizeof(CREC), 1, fout);
        old = cr[a];
    }
    fwrite(&old, sizeof(CREC), 1, fout);
    return 0;
}

/* Check if two cooccurrence records are for the same two words, used for qsort */
int compare_crec(const void *a, const void *b) {
    int c;
    //if (cmp_times==twoexp){
        //fprintf(stderr,"cmp %d: a.word1=%d a.word2=%d b.word1=%d b.word2=%d\n",cmp_times,((CREC *) a)->word1,((CREC *) a)->word2,((CREC *) b)->word1,((CREC *) b)->word2);
        //twoexp*=2;
    //}
    //cmp_times++;
    if( (c = ((CREC *) a)->word1 - ((CREC *) b)->word1) != 0) return c;
    else return (((CREC *) a)->word2 - ((CREC *) b)->word2);
    
}

/* Check if two cooccurrence records are for the same two words */
int compare_crecid(CRECID a, CRECID b) {
    int c;
    if( (c = a.word1 - b.word1) != 0) return c;
    else return a.word2 - b.word2;
}

/* Swap two entries of priority queue */
void swap_entry(CRECID *pq, int i, int j) {
    CRECID temp = pq[i];
    pq[i] = pq[j];
    pq[j] = temp;
}

/* Insert entry into priority queue */
void insert(CRECID *pq, CRECID new, int size) {
    int j = size - 1, p;
    pq[j] = new;
    while( (p=(j-1)/2) >= 0 ) {  //j>=1; it's a binary heap!
        if(compare_crecid(pq[p],pq[j]) > 0) {swap_entry(pq,p,j); j = p;} //if p's rank>j's (1st word, if equal then 2nd word), swap
        else break;
    }
}

/* Delete entry from priority queue */
void delete(CRECID *pq, int size) {
    int j, p = 0;
    pq[p] = pq[size - 1]; //pq[0]=pq[size - 1], delete the root
    while( (j = 2*p+1) < size - 1 ) {
        if(j == size - 2) {
            if(compare_crecid(pq[p],pq[j]) > 0) swap_entry(pq,p,j);
            return;
        }
        else {
            if(compare_crecid(pq[j], pq[j+1]) < 0) {
                if(compare_crecid(pq[p],pq[j]) > 0) {swap_entry(pq,p,j); p = j;}
                else return;
            }
            else {
                if(compare_crecid(pq[p],pq[j+1]) > 0) {swap_entry(pq,p,j+1); p = j + 1;}
                else return;
            }
        }
    }
}

/* Write top node of priority queue to file, accumulating duplicate entries */
int merge_write(CRECID new, CRECID *old, FILE *fout) {
    if(new.word1 == old->word1 && new.word2 == old->word2) {
        old->val += new.val;
        return 0; // Indicates duplicate entry
    }
    fwrite(old, sizeof(CREC), 1, fout);
    *old = new; //write "new" data to "old" address
    return 1; // as a flag to show it's Actually wrote to file
}

/* Merge [num] sorted files of cooccurrence records */
int merge_files(int num) {
    int i, size;
    long long counter = 0;
    CRECID *pq, new, old;
    char filename[200];
    FILE **fid, *fout;
    fid = malloc(sizeof(FILE) * num);
    pq = malloc(sizeof(CRECID) * num);
    fout = stdout;
    if(verbose > 1) fprintf(stderr, "Merging cooccurrence files: processed 0 lines.");
    
    /* Open all files and add first entry of each to priority queue */
    for(i = 0; i < num; i++) {
        sprintf(filename,"%s_%04d.bin",file_head,i); //start from overflow_0000.bin
        fid[i] = fopen(filename,"rb");
        if(fid[i] == NULL) {fprintf(stderr, "Unable to open file %s.\n",filename); return 1;}
        fread(&new, sizeof(CREC), 1, fid[i]); //copy the 1st CREC data in file_i to a CRECID tmp data
        new.id = i; //set the id as the file num -- i
        insert(pq,new,i+1); //heap implemented priority queue
    }
    
    /* Pop top node, save it in old to see if the next entry is a duplicate */
    size = num;
    old = pq[0]; //top of the heap
    i = pq[0].id; //record file_i
    delete(pq, size);
    fread(&new, sizeof(CREC), 1, fid[i]); //continue to read file_i's next record
    if(feof(fid[i])) size--;
    else {
        new.id = i;
        insert(pq, new, size);
    }
    
    /* Repeatedly pop top node and fill priority queue until files have reached EOF */
    while(size > 0) {
        counter += merge_write(pq[0], &old, fout); // Only count the lines written to file, not duplicates
        if((counter%INTERV) == 0) if(verbose > 1) fprintf(stderr," %lldlines.",counter);
        i = pq[0].id;
        delete(pq, size);
        fread(&new, sizeof(CREC), 1, fid[i]);
        if(feof(fid[i])) size--;
        else {
            new.id = i;
            insert(pq, new, size);
        }
    }
    fwrite(&old, sizeof(CREC), 1, fout);
    fprintf(stderr," Merging cooccurrence files: processed %lld lines.\n",++counter);
    //for(i=0;i<num;i++) { 
        //sprintf(filename,"%s_%04d.bin",file_head,i);
        //remove(filename);
    //}
    fprintf(stderr,"\n");
    return 0;
}

/* Collect word-word cooccurrence counts from input stream */
int get_cooccurrence() {
    int flag, x, y, fidcounter = 1;
    long long a, j = 0, k, id, counter = 0, ind = 0, vocab_size, w1, w2, *lookup, *history, bigram_size, tp;
    char format[20], filename[200], str[MAX_STRING_LENGTH + 1];
    FILE *fid, *foverflow;
    real *bigram_table, r;
    HASHREC *htmp, **vocab_hash = inithashtable(); //2^20 HASHRECs
    CREC *cr = malloc(sizeof(CREC) * (overflow_length + 1)); //cr for overflow
    if (cr==NULL) {fprintf(stderr, "can't allocate mem for cr!\n"); return 1;}
    history = malloc(sizeof(long long) * window_size);
    if (history==NULL) {fprintf(stderr, "can't allocate mem for history!\n"); return 1;}
    
    fprintf(stderr, "COUNTING COOCCURRENCES\n");
    if(verbose > 0) {
        fprintf(stderr, "window size: %d\n", window_size);
        if(symmetric == 0) fprintf(stderr, "context: asymmetric\n");
        else fprintf(stderr, "context: symmetric\n");
    }
    if(verbose > 1) fprintf(stderr, "max product: %lld\n", max_product);
    if(verbose > 1) fprintf(stderr, "overflow length: %lld\n", overflow_length);
    sprintf(format,"%%%ds %%lld", MAX_STRING_LENGTH); // 
    if(verbose > 1) fprintf(stderr, "Reading vocab from file \"%s\"...", vocab_file);
    fid = fopen(vocab_file,"r");
    if(fid == NULL) {fprintf(stderr,"Unable to open vocab file %s.\n",vocab_file); return 1;}
	 // Here id is not used: inserting vocab words into hash table with their frequency rank, j
    while(fscanf(fid, format, str, &id) != EOF) hashinsert(vocab_hash, str, ++j); //str:word id:frequency
    fclose(fid);
    vocab_size = j;
    j = 0;
    if(verbose > 1) fprintf(stderr, "loaded %lld words.\nBuilding lookup table...", vocab_size);
    
    /* Build auxiliary lookup table used to index into bigram_table */
    lookup = (long long *)calloc( vocab_size+1 , sizeof(long long) ); //origin: no "+1"
    if (lookup == NULL) {
        fprintf(stderr, "Couldn't allocate memory for lookup!");
        return 1;
    }
    lookup[0] = 1;
    for(a = 1; a <= vocab_size; a++) {  //max_product~=14million
        if((lookup[a] = max_product / a) < vocab_size) lookup[a] += lookup[a-1];  //vocab_size~=72k, its square ~= 5billion
        else lookup[a] = lookup[a-1] + vocab_size;
    }//1+k*vocab_size+max_product/(k+1)+max_product/(k+2)...+max_product/(vocab_size), notice at last a=vocab_size+1
	//fprintf(stderr, "\nlookup[12]=%lld\n",lookup[12]);
    bigram_size=lookup[a-1];
    if(verbose > 1) fprintf(stderr, "table contains %lld elements (bigram_table[] size).\n",bigram_size);
    
    /* Allocate memory for full array which will store all cooccurrence counts for words whose product of frequency ranks is less than max_product */
    bigram_table = (real *)calloc( bigram_size , sizeof(real) );
    if (bigram_table == NULL) {
        fprintf(stderr, "Couldn't allocate memory for bigram_table!");
        return 1;
    }
    
    fid = stdin; //<text8>
    sprintf(format,"%%%ds",MAX_STRING_LENGTH);
    sprintf(filename,"%s_%04d.bin",file_head, fidcounter);  //overflow_0001.bin, file_head='overflow'; output is written into filename[]
    foverflow = fopen(filename,"w");//generate overflow_0001.bin
    if (foverflow==NULL) {fprintf(stderr, "can't open %s!\n",filename); return 1;}
    if(verbose > 1) fprintf(stderr,"Processing token: 0");
    
    /* For each token in input stream, calculate a weighted cooccurrence sum within window_size */
    while (1) { //fprintf(stderr,"\nj=%d",j);//wrong on the 4th loop
        if(ind >= overflow_length - window_size) { // If overflow buffer is (almost) full, sort it and write it to temporary file
            qsort(cr, ind, sizeof(CREC), compare_crec);
            write_chunk(cr,ind,foverflow);  //store in disk
            fclose(foverflow);  //close it
            fidcounter++;
            sprintf(filename,"%s_%04d.bin",file_head,fidcounter); //set the file name
            foverflow = fopen(filename,"w");  //generate a new empty file with the name
            if (foverflow==NULL) {fprintf(stderr, "can't open %s!\n",filename); return 1;}
            ind = 0;
        }
        flag = get_word(str, fid);
        if(feof(fid)) break;
        if(flag == 1) {j = 0; continue;} // Newline, reset line index (j)
        counter++;
        if((counter%INTERV) == 0) if(verbose > 1) fprintf(stderr," %lld ",counter); //check!
        htmp = hashsearch(vocab_hash, str);
        if (htmp == NULL) continue; // Skip out-of-vocabulary words
        w2 = htmp->id; // Target word (frequency rank)
		  //fprintf(stderr,"\nj=%lld",j);
        for(k = j - 1; k >= ((j > window_size) ? j - window_size : 0); k--) { //fprintf(stderr,"\nk=%lld;j=%lld",k,j);
		  // Iterate over all words to the left of target word, but not past beginning of line, only when j>=15 the for-loop executes
            w1 = history[k % window_size]; // Context word (frequency rank)
            if ( w1 < max_product/w2 ) { //fprintf(stderr,"\nw1=%lld;w2=%lld",w1,w2);
				// Product (of the 2 freq_ranks) is small (high freq) enough to store in a full array
                tp=lookup[w1-1]+w2-2; if ((tp>bigram_size-1)||(tp<0)) {fprintf(stderr,"bigram_size=%lld\n",tp); return 1;}
                bigram_table[tp] += 1.0/((real)(j-k)); // Weight by inverse of distance between words
					 //fprintf(stderr,"\nlookup[w1-1]=%lld;j-k=%lld;bigram_table[]=%f",lookup[w1-1],j-k,bigram_table[lookup[w1-1] + w2 - 2]);
					 //fprintf(stderr,"\nlookup[w2-1]=%lld;j-k=%lld",lookup[w2-1],j-k);
					 //fprintf(stderr,"\nbigram_table[]=%f",bigram_table[lookup[w2-1] + w1 - 2]);
                if(symmetric > 0){
                    tp=lookup[w2-1]+w1-2; if ((tp>bigram_size-1)||(tp<0)) {fprintf(stderr,"bigram_size=%lld\n",tp); return 1;}
                    bigram_table[lookup[w2-1] + w1 - 2] += 1.0/((real)(j-k));
                } // If symmetric context is used, exchange roles of w2 and w1 (ie look at right context too)
					
            }
            else { // Product is too big, data is likely to be sparse. Store these entries in a temporary buffer to be sorted, merged (accumulated), and written to file when it gets full.
                cr[ind].word1 = w1;
                cr[ind].word2 = w2;
                cr[ind].val = 1.0/((real)(j-k));
                ind++; // Keep track of how full temporary buffer is
                //fprintf(stderr,"ind=%lld,w1=%lld,w2=%lld,j=%lld,k=%lld\n",ind-1,w1,w2,j,k);
                if(symmetric > 0) { // Symmetric context
                    cr[ind].word1 = w2;
                    cr[ind].word2 = w1;
                    cr[ind].val = 1.0/((real)(j-k));
                    ind++;
                }
                
            }
        }
        history[j % window_size] = w2; // Target word is stored in circular buffer to become context word in the future
        j++;
    }
    
    /* Write out temp buffer for the final time (it may not be full) */
    if(verbose > 1) fprintf(stderr," Processed %lld tokens.\n",counter);
    if (ind>0){ fprintf(stderr,"ind=%lld\n",ind);
        qsort(cr, ind, sizeof(CREC), compare_crec); //fprintf(stderr,"qsort ok!\n");
        write_chunk(cr,ind,foverflow);
    } 
    sprintf(filename,"%s_0000.bin",file_head); //the last one is named overflow_0000.bin
    
    /* Write out full bigram_table, skipping zeros */
    if(verbose > 1) fprintf(stderr, "Writing cooccurrences to disk");
    fid = fopen(filename,"w"); //generate overflow_0000.bin
    if (fid==NULL) {fprintf(stderr, "can't open %s!\n",filename); return 1;}
    j = 1e6;
    for(x = 1; x <= vocab_size; x++) {
        if( (long long) (0.75*log(vocab_size / x)) < j) {j = (long long) (0.75*log(vocab_size / x)); if(verbose > 1) fprintf(stderr,".");} // log's to make it look (sort of) pretty
        for(y = 1; y <= (lookup[x] - lookup[x-1]); y++) {
            if((r = bigram_table[lookup[x-1] - 2 + y]) != 0) {
                fwrite(&x, sizeof(int), 1, fid);
                fwrite(&y, sizeof(int), 1, fid);
                fwrite(&r, sizeof(real), 1, fid);
            } //write them into overflow_0000.bin
        }
    }
    
    if(verbose > 1) fprintf(stderr,"%d files in total.\n",fidcounter + 1); fprintf(stderr, "so far so good!\n");
    fclose(fid); fprintf(stderr, "fclose(fid)\n");
    fclose(foverflow);
    free(cr);
    free(lookup);
    free(bigram_table);
    free(vocab_hash);
    return merge_files(fidcounter + 1); // Merge the sorted temporary files
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
    real rlimit, n = 1e6;
    vocab_file = malloc(sizeof(char) * MAX_STRING_LENGTH);
    file_head = malloc(sizeof(char) * MAX_STRING_LENGTH);
    
    if (argc == 1) {
        printf("Tool to calculate word-word cooccurrence statistics\n");
        printf("Author: Jeffrey Pennington (jpennin@stanford.edu)\n\n");
        printf("Usage options:\n");
        printf("\t-verbose <int>\n");
        printf("\t\tSet verbosity: 0, 1, or 2 (default)\n");
        printf("\t-symmetric <int>\n");
        printf("\t\tIf <int> = 0, only use left context; if <int> = 1 (default), use left and right\n");
        printf("\t-window-size <int>\n");
        printf("\t\tNumber of context words to the left (and to the right, if symmetric = 1); default 15\n");
        printf("\t-vocab-file <file>\n");
        printf("\t\tFile containing vocabulary (truncated unigram counts, produced by 'vocab_count'); default vocab.txt\n");
        printf("\t-memory <float>\n");
        printf("\t\tSoft limit for memory consumption, in GB -- based on simple heuristic, so not extremely accurate; default 4.0\n");
        printf("\t-max-product <int>\n");
        printf("\t\tLimit the size of dense cooccurrence array by specifying the max product <int> of the frequency counts of the two cooccurring words.\n\t\tThis value overrides that which is automatically produced by '-memory'. Typically only needs adjustment for use with very large corpora.\n");
        printf("\t-overflow-length <int>\n");
        printf("\t\tLimit to length <int> the sparse overflow array, which buffers cooccurrence data that does not fit in the dense array, before writing to disk. \n\t\tThis value overrides that which is automatically produced by '-memory'. Typically only needs adjustment for use with very large corpora.\n");
        printf("\t-overflow-file <file>\n");
        printf("\t\tFilename, excluding extension, for temporary files; default overflow\n");

        printf("\nExample usage:\n");
        printf("./cooccur -verbose 2 -symmetric 0 -window-size 10 -vocab-file vocab.txt -memory 8.0 -overflow-file tempoverflow < corpus.txt > cooccurrences.bin\n\n");
        return 0;
    }
	
    if ((i = find_arg((char *)"-verbose", argc, argv)) > 0) verbose = atoi(argv[i + 1]);
    if ((i = find_arg((char *)"-symmetric", argc, argv)) > 0) symmetric = atoi(argv[i + 1]);
    if ((i = find_arg((char *)"-window-size", argc, argv)) > 0) window_size = atoi(argv[i + 1]);
    if ((i = find_arg((char *)"-vocab-file", argc, argv)) > 0) strcpy(vocab_file, argv[i + 1]);
    else {strcpy(vocab_file, (char *)"vocab.txt"); fprintf(stderr,"Use vocab.txt!");}
    if ((i = find_arg((char *)"-overflow-file", argc, argv)) > 0) strcpy(file_head, argv[i + 1]);
    else strcpy(file_head, (char *)"overflow");
    if ((i = find_arg((char *)"-memory", argc, argv)) > 0) memory_limit = atof(argv[i + 1]);
	 //mingw32 cannot use >4G, mingw64,12G,mem used at most 5.1G(probably should -2.5G for other progs)
	 //since the total nonzero data for text8 is <1G, that may be still a waste of mem.
    
    /* The memory_limit determines a limit on the number of elements in bigram_table and the overflow buffer */
    /* Estimate the maximum value that max_product can take so that this limit is still satisfied */
    rlimit = 0.85 * (real)memory_limit * 1073741824/(sizeof(CREC));
    while(fabs(rlimit - n * (log(n) + 0.1544313298)) > 1e-3) n = rlimit / (log(n) + 0.1544313298); fprintf(stderr,"n: %f\n", n);
    max_product = (long long) n; //nlogn=0.85*4G/16Bytes ~= 0.228billion, then n~=14M
    overflow_length = (long long) rlimit/6; // 0.85 + 1/6 ~= 1
    
    /* Override estimates by specifying limits explicitly on the command line */
    if ((i = find_arg((char *)"-max-product", argc, argv)) > 0) max_product = atoll(argv[i + 1]);
    if ((i = find_arg((char *)"-overflow-length", argc, argv)) > 0) overflow_length = atoll(argv[i + 1]);
    
	 //i = get_cooccurrence();
	 //fprintf(stderr," files deleted: %d, %d\n", remove("overflow_0000.bin"), remove("overflow_0001.bin") );
    return get_cooccurrence();
}


