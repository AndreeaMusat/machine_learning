run_mini_map:
	python q_learning_skel.py --train_episodes 1500 \
							  --final_show \
							  --plot \
							  --eval_episodes 5 \
							  --learning_rate 0.9 \
							  --discount 0.99 \
							  --map_file mini_map \
							  --epsilon 0.08

run_big_map:
	python q_learning_skel.py --train_episodes 10000 \
							  --final_show \
							  --plot \
							  --eval_episodes 10 \
							  --learning_rate 0.8 \
							  --discount 0.99 \
							  --map_file big_map \
							  --epsilon 0.05

run_huge_map:
	python q_learning_skel.py --train_episodes 10000 \
							  --final_show \
							  --plot \
							  --eval_episodes 10 \
							  --learning_rate 0.5 \
							  --discount 0.8 \
							  --map_file huge_map \
							  --epsilon 0.005