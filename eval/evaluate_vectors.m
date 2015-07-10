addpath('./eval');
vocab_file = 'vocab.txt';
vectors_file = 'vectors.bin';


fid = fopen(vocab_file, 'r');
words = textscan(fid, '%s %f'); %%word & frequency(double)
fclose(fid);
words = words{1}; %%only keep words
vocab_size = length(words);
global wordMap
wordMap = containers.Map(words(1:vocab_size),1:vocab_size);%%word->rank

fid = fopen(vectors_file,'r');
fseek(fid,0,'eof');
vector_size = ftell(fid)/16/vocab_size - 1;%%vectors.bin: 8*51*2*vocab_size
%%if vectors.txt, there're vocab_size of (diff length) 103 strings, 1 word, 102 numbers
frewind(fid);
WW = fread(fid, [vector_size+1 2*vocab_size], 'double')'; %%the matrix is (2*vocab_size,vector_size+1)
fclose(fid); 

W1 = WW(1:vocab_size, 1:vector_size); % word vectors
W2 = WW(vocab_size+1:end, 1:vector_size); % context (tilde) word vectors

W = W1 + W2; %Evaluate on sum of word vectors %%(vocab_size,vector_size)
W = bsxfun(@rdivide,W,sqrt(sum(W.*W,2))); %normalize vectors before evaluation
%%for each word (vocab_size numbers): every vector is on a vector_size-dim unit ball
%%so 1-W*W'=>cosine similarity between every pair of the words
evaluate_vectors(W);
%%exit

