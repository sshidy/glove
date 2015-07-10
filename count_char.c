#include <stdio.h>

#define INTERV 10000000

char* filename="glove.6B.300d.txt"; //glove.6B.300d.txt

int main()
{
    char c, c0;
    unsigned int ct_space=0, ct_tab=0, ct_line=0;
    unsigned long long ct=0;

    if (freopen(filename,"r",stdin)!=NULL){
        while ((c = getchar()) != EOF){
            ct++;
            if (ct%INTERV==0) fprintf(stderr, "%llu chars. ", ct);
            c0=c;
            switch (c) {
                case ' ': ct_space++;break;
                case '\t': ct_tab++;break;
                case '\n': ct_line++;break;
            }
//            if (c=='\n') ct_line++; //only count the line before this '\n'
        }
        if (c0!='\n') ct_line++; //if the last char (before EOF) is not a newline, the last line should be counted
        fprintf(stderr, "%u lines, %u tabs, %u white spaces.\n", ct_line, ct_tab, ct_space);
    }else{
        fprintf(stderr, "fail to read!\n");
    }

//    freopen("glove_vecs.bin","w",stdout);

    return 0;
}
