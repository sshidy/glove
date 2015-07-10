CC = gcc
#For older gcc, use -O3 or -O2 instead of -Ofast
CFLAGS = -lm -pthread -Ofast -march=native -funroll-loops -Wno-unused-result
COFLAGS = -lm -Ofast -march=native -funroll-loops -Wno-unused-result
# -lm: The -l means link a library. The m means the math library, the standard library that has common math functions like sqrt, sin, cos, log, etc.
# -Ofast: opt for speed; -march= target cpu; 
# -Wno-unused-result: Do not Warn if a caller of a function marked with attribute warn_unused_result (see Function Attributes) does not use its return value. The default is -Wunused-result.
all: glove shuffle cooccur vocab_c
others: shuffle cooccur vocab_c

glove : glove.c
	$(CC) glove.c -o glove $(CFLAGS)
shuffle : shuffle.c
	$(CC) shuffle.c -o shuffle $(COFLAGS)
cooccur : cooccur.c
	$(CC) cooccur.c -o cooccur $(COFLAGS)
vocab_c : vocab_count.c
	$(CC) vocab_count.c -o vocab_count $(COFLAGS)

clean:
	rm -rf glove shuffle cooccur vocab_count
