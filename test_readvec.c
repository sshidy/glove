#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//#define INTERV 10000
#define MAX_STRING_LEN 200 //have to be large, otherwise unexpected mistakes when long "--"s
#define VEC_DIM 300
#define VOCAB_NUM 400000

const char* input_file="glove.6B.300d.txt";
const char* vocab_file="vocab_g.txt";
const char* vector_file="vectors_g.bin";

int main()
{
    char c, c0, s[MAX_STRING_LEN];
    unsigned int ct_space=0, i, ct_line=0;
    FILE *fid_vocab, *fid_vector;
    double *W;

    if (freopen(input_file,"r",stdin)!=NULL){
        fid_vocab=fopen(vocab_file,"w");
        fid_vector=fopen(vector_file,"wb");
        W=malloc(VOCAB_NUM*VEC_DIM*sizeof(double));
        while ((c = getchar()) != EOF){
            c0=c;
            switch (c) {
                case ' ':
                    scanf("%s",s);
                    W[ct_space]=atof(s);
                    ct_space++;
                    break;
                case '\n':
                    ct_line++;
                    break;
                default:
                    ungetc(c,stdin);
                    scanf("%s",s);
                    fprintf(fid_vocab,"%s\n", s);
                    if (strlen(s)>30)  fprintf(stderr, "%u %s\n", ct_line, s);

            }
        }
        for (i=0;i<ct_space;i++) fwrite(&W[i],sizeof(double),1,fid_vector);
        if (c0!='\n') ct_line++; //if the last char (before EOF) is not a newline, the last line should be counted
        fprintf(stderr, "\n%u lines, %u spaces.\n", ct_line, ct_space);
    }else{
        fprintf(stderr, "fail to read!\n");
    }

    if (ct_space==VEC_DIM*VOCAB_NUM) fprintf(stderr, "VEC_DIM*VOCAB_NUM matches the spaces.\n");
    else fprintf(stderr, "Warning: VEC_DIM*VOCAB_NUM does not match the spaces!\n");

    return 0;
}
