# glove
test and edit the Glove code

We test and edit the Glove code ( http://nlp.stanford.edu/projects/glove/ ) for word embeddings.

The original paper is "GloVe: Global Vectors for Word Representation" on EMNLP2014. The original code is from http://www-nlp.stanford.edu/software/glove.tar.gz (question data can be obtained from it)

Glove is more efficient than CBOW and Skip-Gram that run on word2vec( https://code.google.com/p/word2vec/ ). It is also more practically efficient than some recent progress in the form of matrix factorization in 2015. Thanks to its authors.

Many lines are edited to test and to rebuild. The two most important modifications are made on the cooccur.c, both in "int get_cooccurrence()":

- "lookup = (long long *)calloc( vocab_size , sizeof(long long) );" is corrected as  
  "lookup = (long long *)calloc( vocab_size + 1, sizeof(long long) );"

One could see why in the following "for" loop. 
This bug can cause "core dump" at unexpected execution time on various configurations.

- The last "qsort(cr, ind, sizeof(CREC), compare_crec); write_chunk(cr,ind,foverflow);" should be under condition "if (ind>0)"

This bug is possible to inject a record with cr.word1=0, cr.word2=0, cr.val=0.0 into the cooccurance bin file if the co-occurance data is relatively small according to the "-memery" set. It's hard to find it out after shuffling. It can cause glove.c crash after computing log(cr.val) and AdaGrad (divde 0).
	

The test is successful on Ubuntu 14.04. 
Although the c files can be compiled by MinGW-64 on Windows, the binary file generated by cooccur.c under MinGW-64 is a mess.
