addpath('./eval');
vocab_file = 'vocab_g.txt';
vectors_file = 'vectors_g.bin';

fid = fopen(vocab_file, 'r');
words = textscan(fid, '%s'); %%word
fclose(fid);
words = words{1};
vocab_size = length(words)%;
global wordMap
wordMap = containers.Map(words(1:vocab_size),1:vocab_size);%%word->rank

fid = fopen(vectors_file,'r');
fseek(fid,0,'eof');
vector_size = ftell(fid)/8/vocab_size%;%%vectors.bin: 8*300*vocab_size
frewind(fid);
W = fread(fid, [vector_size vocab_size], 'double')'; %%the matrix is (vocab_size,vector_size)
fclose(fid); 

W = bsxfun(@rdivide,W,sqrt(sum(W.*W,2))); %normalize vectors before evaluation
%%for each word (vocab_size numbers): every vector is on a vector_size-dim unit ball
%%so 1-W*W'=>cosine similarity between every pair of the words
evaluate_vectors(W);
%%exit

