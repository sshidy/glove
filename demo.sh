#!/bin/bash

make

CORPUS=text9
VOCAB_FILE=vocab.txt
COOC_FILE=cooc.bin
COOC_SHUF_FILE=cooc.shuf.bin
SAVE_FILE=vectors
VERBOSE=2
MEMORY=8.0
VOCAB_MIN_COUNT=5
VECTOR_SIZE=200
MAX_ITER=15
WINDOW_SIZE=10
BINARY=2
NUM_THREADS=8
X_MAX=100

./vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $CORPUS > $VOCAB_FILE
if [[ $? -eq 0 ]]  ##[ "$nval" -eq 0 ]  Integer test; true if equal to 0. $? gives the exit status of the last command that was executed. This should be 0 if the command exited normally.
  then
  ./cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE < $CORPUS > $COOC_FILE
  if [[ $? -eq 0 ]]
  then
    ./shuffle -memory $MEMORY -verbose $VERBOSE < $COOC_FILE > $COOC_SHUF_FILE
    if [[ $? -eq 0 ]]
    then
       ./glove -save-file $SAVE_FILE -threads $NUM_THREADS -input-file $COOC_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE
       if [[ $? -eq 0 ]]
       then
	   matlab -nodisplay -nodesktop -nojvm -nosplash < ./eval/read_and_evaluate.m 1>&2 
       fi
    fi
  fi
fi


