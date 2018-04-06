RANDOM_AGENT=--left-strategy random
GREEDY_AGENT=--left-strategy greedy
EGREEDY_AGENT=--left-strategy e-greedy

RANDOM_OPPONENT=--right-strategy random
GREEDY_OPPONENT=--right-strategy greedy
ALMOST_PERFECT_OPPONENT=--right-strategy almost-perfect

SMALL_WIDTH=--width 140
SMALL_HEIGHT=--height 100
SMALL_STEP=--step 10
SMALL_CONFIG=$(SMALL_WIDTH) $(SMALL_HEIGHT) $(SMALL_STEP)

BIG_WIDTH=--width 345 
BIG_HEIGHT=--height 285
BIG_STEP=--step 15
BIG_CONFIG=$(BIG_WIDTH) $(BIG_HEIGHT) $(BIG_STEP)

run_small_greedy_greedy:
	python pong.py $(SMALL_CONFIG) \
				   $(GREEDY_AGENT) \
				   $(GREEDY_OPPONENT) \
				   --learning-rate 0.99 \
				   --discount-factor 0.6 \
				   --output-file junk_file

run_small_random_random:
	python pong.py $(SMALL_CONFIG) \
	 			   $(RANDOM_AGENT) \
	 			   $(RANDOM_OPPONENT) \
	 			   --output-file junk_file

run_small_egreedy_almost_perfect:
	python pong.py $(SMALL_CONFIG) \
				   $(EGREEDY_AGENT) \
				   $(ALMOST_PERFECT_OPPONENT) \
				   --learning-rate 0.95 \
				   --discount-factor 0.6 \
				   --epsilon 0.01 \
				   --output-file junk_file

run_big_greedy_greedy:
	python pong.py $(BIG_CONFIG) \
				   $(GREEDY_AGENT) \
				   $(GREEDY_OPPONENT) \
				   --learning-rate 0.98 \
				   --discount-factor 0.5 \
				   --output-file junk_file

run_big_greedy_almost_perfect:
	python pong.py $(BIG_CONFIG) \
				   $(GREEDY_AGENT) \
				   $(ALMOST_PERFECT_OPPONENT) \
				   --learning-rate 1.0 \
				   --discount-factor 0.6 \
				   --output-file junk_file
