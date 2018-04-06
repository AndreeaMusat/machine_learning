# Andreea Musat, March 2018

import pygame
import random
import sys
import argparse
import time
import math
import pickle
import numpy as np
from pygame import Rect
from static_info import *
from matplotlib import pyplot as plt

class pongGame():

	def __init__(self, args):
		self.width = args.width
		self.height = args.height
		self.out_file = args.output_file
		self.step = args.step
		self.score = [0, 0]

		self.restart_game()
		self.init_gui()
		self.init_learning_models(args)

	#################################### GAME LOGIC #############################
	def restart_game(self):
		top = int((self.height - 2 * MARGIN) / 2 + MARGIN - PAD_HEIGHT / 2)
		self.left_pad = Rect(PAD_OFFSET, top, PAD_WIDTH, PAD_HEIGHT)
		self.right_pad = Rect(self.width - MARGIN, top, PAD_WIDTH, PAD_HEIGHT)

		left = int((self.width - 2 * MARGIN) / 2 + MARGIN - BALL_SIZE / 2)
		top = int((self.height - 2 * MARGIN) / 2 + MARGIN - BALL_SIZE / 2)

		self.ball_pos = Rect(left, top, BALL_SIZE, BALL_SIZE)
		self.ball_dir = random.choice(BALL_DIRS)
		self.round_over = False
		self.left_curr_score = 0

	def move_ball(self):
		self.ball_pos.x += self.ball_dir[0]
		self.ball_pos.y += self.ball_dir[1]

	def check_collision(self):
		self.x_collision = False
		self.y_collision = False

		if self.ball_pos.top == MARGIN or self.ball_pos.bottom >= self.height - MARGIN:
			self.x_collision = True

		if self.ball_pos.left == MARGIN or self.ball_pos.right >= self.width - MARGIN:
			self.y_collision = True

	def check_ball_caught(self):
		self.left_reward = NO_REWARD
		self.right_reward = NO_REWARD

		if self.ball_pos.left == MARGIN and \
		  (self.ball_pos.bottom <= self.left_pad.top or \
		   self.ball_pos.top >= self.left_pad.bottom):
			self.y_collision = False
			self.left_reward = MINUS_REWARD
			self.round_over = True
			self.score[1] += 1
			if DEBUG:
				print("GOT MINUS REWARD")

		if self.y_collision and self.ball_pos.left == MARGIN:
			self.left_reward = PLUS_REWARD
			self.left_curr_score += 1
			if DEBUG:
				print("GOT PLUS REWARD")

		
		if self.ball_pos.right == self.width - MARGIN and \
		  (self.ball_pos.bottom <= self.right_pad.top or \
		   self.ball_pos.top >= self.right_pad.bottom):
			self.y_collision = False
			self.right_reward = MINUS_REWARD
			self.round_over = True
			self.score[0] += 1
			if DEBUG:
				print("right got MINUS reward")


		if self.y_collision and self.ball_pos.right == self.width - MARGIN:
			self.right_reward = PLUS_REWARD
			if DEBUG:
				print("right got PLUS reward")

		# make sure the learning modules have the rewards
		self.update_rewards_learning_model()

	def update_ball_dir(self):
		if self.x_collision:
			self.ball_dir = (self.ball_dir[0], -self.ball_dir[1])

		if self.y_collision:
			self.ball_dir = (-self.ball_dir[0], self.ball_dir[1])

		self.x_collision = False
		self.y_collision = False

	#################################### END GAME LOGIC ############################

	#################################### LEARNING ##################################
	def init_learning_models(self, args):

		self.learning_rate = args.learning_rate
		self.discount_factor = args.discount_factor
		self.epsilon = args.epsilon

		if args.resume_left == None:
			self.left_model = {}
			self.left_model['strategy'] = args.left_strategy

			if self.left_model['strategy'] in ['greedy', 'e-greedy']:
				self.left_model['max_i'] = int((self.height - 2 * MARGIN) / self.step)
				self.left_model['max_j'] = int((self.width - 2 * MARGIN) / self.step)
				self.init_Q('left')

				# this is used for updating Q; must use 'left_' in order to be able
				# to share parameters between models (should have a different set
				# of prev_state/action/reward for the right player)
				self.left_model['left_prev_state'] = None
				self.left_model['left_prev_action'] = None
				self.left_model['left_prev_reward'] = None

				# this is used for decaying the learning rate
				self.left_model['cnt'] = 0

			if self.left_model['strategy'] == 'e-greedy':
				self.left_model['tried'] = {}
		
		self.right_model = {}
		self.right_model['strategy'] = args.right_strategy

		if DEBUG:
			for k in self.left_model:
				print(k)

		

	def restart_learning_module(self):

		self.left_model['left_prev_state'] = None
		self.left_model['left_prev_action'] = None
		self.left_model['left_prev_reward'] = None


	def init_Q(self, player_name):

		model = self.left_model if player_name == 'left' else 'right'
		
		model['Q'] = {}
		for player_i in range(model['max_i']):
				for ball_i in range(model['max_i']):
					for ball_j in range(model['max_j']):
						for ball_dir_idx in range(NUM_BALL_DIRS):
							for action in ACTIONS:
								# r = random.uniform(MINUS_REWARD, PLUS_REWARD)
								state = (player_i, (ball_i, ball_j), ball_dir_idx)
								model['Q'][state, action] = random.random()


	def rect_to_coords(self, rectangle, step):
		center_x = rectangle.top + rectangle.height / 2
		i_coord = int((center_x - MARGIN) / step)
		center_y = rectangle.left + rectangle.width / 2
		j_coord = int((center_y - MARGIN) / step)
		return i_coord, j_coord

	# return the state (as it is in the learning model) corresponding to 
	# current game
	def get_state(self, player_name):

		# state = player_i, (ball_i, ball_j), ball_dir_idx
		player_pad = self.left_pad if player_name == 'left' else self.right_pad

		player_i, player_j = self.rect_to_coords(player_pad, self.step)
		ball_i, ball_j = self.rect_to_coords(self.ball_pos, self.step)
		ball_dir_idx = BALL_DIRS.index(self.ball_dir)
		state = (player_i, (ball_i, ball_j), ball_dir_idx)
		
		if player_name == 'left':
			return state

		# this is a special case; must reflect the coordinates
		if player_name == 'right':
			if DEBUG:
				print('-------------------------------------------')
				print('BEFORE: ', state)
				print('right pad top = ', self.right_pad.top)
				print('max j = ', self.left_model['max_j'])

			ball_j = self.left_model['max_j'] - ball_j - 1
			ball_dir_idx = BALL_DIRS.index((-self.ball_dir[0], self.ball_dir[1]))
			state = (player_i, (ball_i, ball_j), ball_dir_idx)
			
			if DEBUG:
				print('AFTER:', state)
				print('-------------------------------------------')

			return state

	def get_available_actions(self, player_name):
		actions = ['nop']

		pad = self.left_pad if player_name == 'left' else self.right_pad

		if pad.top > MARGIN:
			actions.append('up')

		if pad.bottom < self.height - MARGIN:
			actions.append('down')

		return actions

	# truly returns the best action if model is greedy; otherwise 
	# returns a random action with probability epsilon and the best action
	# with probability 1 - epsilon
	def get_best_action(self, player_name, state, available_actions):
		model = self.left_model

		best_action = None
		best_action_utility = None

		# random.shuffle(available_actions)
		for action in available_actions:
			if best_action_utility is None or \
			   model['Q'][state, action] > best_action_utility:
				best_action_utility = model['Q'][state, action]
				best_action = action

		if action == None:
			print("IN get_best_action, action is None idk what happened")
			sys.exit(-1)

		if DEBUG:
			print("BEST action is ", best_action)

		return best_action


	def decay_learning_rate(self):
		# pass
		if self.learning_rate < 0:
			print("no more learning, alpha = 0")
			return

		self.left_model['cnt'] += 1
		if self.left_model['cnt'] % ITERATIONS == 0:
			self.learning_rate *= math.e**(-K)
			self.epsilon *= 0.99
			self.left_model['cnt'] = 0
			

	# this should return an action and also update the learning model !
	def get_action(self, player_name, do_update=True):

		available_actions = self.get_available_actions(player_name)
		model = self.left_model if player_name == 'left' else self.right_model

		# if strategy is random, return random action for both players
		if model['strategy'] == 'random':
			return random.choice(available_actions)

		# greedy for right player just uses Q learnt by left player 
		if player_name == 'right' and self.right_model['strategy'] == 'greedy':
			curr_state = self.get_state(player_name)
			curr_best_action = self.get_best_action(player_name, curr_state, available_actions)
			return curr_best_action

		# almost perfect strategy: move the pad so that it 'follows' the ball
		if player_name == 'right' and self.right_model['strategy'] == 'almost-perfect':
			if random.random() < EPS:
				return random.choice(available_actions)

			if self.right_pad.centery < self.ball_pos.centery:
				return 'down'
			else:
				return 'up'

		# these are the cases where the model actually learns Q
		if player_name == 'left' and self.left_model['strategy'] in ['greedy', 'e-greedy']:

			model = self.left_model

			if DEBUG:
				print()
				print()
				print("GETTTING ACTION ************************************")			

			curr_state = self.get_state(player_name)

			# curr best action is used for updating the Q value of the previous state and prev action

			curr_best_action = self.get_best_action(player_name, curr_state, available_actions)
			
			if do_update:
				if self.left_model['strategy'] == 'e-greedy':
					not_tried_yet = []
					for action in available_actions:
						if (curr_state, action) not in self.left_model['tried']:
							not_tried_yet.append(action)

					if not_tried_yet != []:
						curr_action = random.choice(not_tried_yet)
					else:
						p = random.random()
						if p < self.epsilon / 10:
							curr_action = random.choice(available_actions)
						else:
							curr_action = curr_best_action

					# mark curr action as tried from curr state
					self.left_model['tried'][curr_state, curr_action] = True
				else:
					curr_action = curr_best_action

			# if i don't want an update, I always want the best action
			else:
				curr_action = curr_best_action

			if DEBUG:
				print("CURR STATE = ", curr_state)
				print("AVAILABLE ACTIONS = ", available_actions)
				print("CURR BEST ACTION = ", curr_best_action)

			if do_update:
				self.decay_learning_rate()

				prefix = player_name + '_'
				# if this is not the first action, update the model
				if model[prefix + 'prev_state'] is not None:
					S_t = model[prefix + 'prev_state']
					a_t = model[prefix + 'prev_action']
					Q_t_old = model['Q'][S_t, a_t]
					alpha = self.learning_rate
					Q_tpp = model['Q'][curr_state, curr_best_action]
					d = self.discount_factor
					r = model[prefix + 'prev_reward']
					Q_t_new = Q_t_old + alpha * (r + d * Q_tpp - Q_t_old)
					model['Q'][S_t, a_t] = Q_t_new

					if DEBUG:
						print('PREV STATE = ', S_t)
						print('PREV ACTION = ', a_t)
						print('PREV REWARD = ', r)
						print('alpha = ', alpha)
						print('discount = ', d)
						print('old Q[s_t, a_t] = ', Q_t_old)
						print('new Q[s_t, a_t] = ', Q_t_new)
						print('Q[s_t+1, a_t+1] = ', Q_tpp)
						print()

				model[prefix + 'prev_state'] = curr_state
				model[prefix + 'prev_action'] = curr_action

			return curr_action

	def update_rewards_learning_model(self):
		self.left_model['left_prev_reward'] = self.left_reward

	def apply_action(self, action, player_name, step):
		if action == 'nop': 
			return

		c = -1 if action == 'up' else 1
		if player_name == 'left':
			self.left_pad.y += c * step

		if player_name == 'right':
			self.right_pad.y += c * step

	####################################### END LEARNING ###########################

	############################### GUI STUFF ######################################
	def init_gui(self):
		pygame.init()
		pygame.font.init()
		pygame.display.set_caption('PONG')

		self.font = pygame.font.SysFont('Arial', 12)

		self.clock = pygame.time.Clock()
		self.screen = pygame.display.set_mode((self.width, self.height))

		self.top_left = (MARGIN, MARGIN)
		self.top_right = (self.width - MARGIN, MARGIN)
		self.bott_left = (MARGIN, self.height - MARGIN)
		self.bott_right = (self.width - MARGIN, self.height - MARGIN)
		self.top_middle = (int(self.width - 2 * MARGIN) / 2 + MARGIN, MARGIN)
		self.bott_middle = (int(self.width - 2 * MARGIN) / 2 + MARGIN, self.height - MARGIN)

	# draw static stuff (bbox lines)
	def static_draw(self):
		pygame.draw.line(self.screen, GRAY, self.top_left, self.top_right)
		pygame.draw.line(self.screen, GRAY, self.top_left, self.bott_left)
		pygame.draw.line(self.screen, GRAY, self.top_right, self.bott_right)
		pygame.draw.line(self.screen, GRAY, self.bott_right, self.bott_left)
		pygame.draw.line(self.screen, GRAY, self.top_middle, self.bott_middle)

	# draw dynamic stuff (player's pads, ball)
	def dynamic_draw(self, phase):
		pygame.draw.rect(self.screen, WHITE, self.left_pad)
		pygame.draw.rect(self.screen, WHITE, self.right_pad)
		pygame.draw.rect(self.screen, WHITE, self.ball_pos)

		# print score, learning rate, discount factor, epsilon
		msg1 = 'Score: ' + str(self.score[0]) + ':' + str(self.score[1])	
		textsurface1 = self.font.render(msg1, False, WHITE)
		textrect1 = textsurface1.get_rect(center=(self.width / 2, MARGIN / 2))
		self.screen.blit(textsurface1, textrect1)

		if phase == 'train' and self.left_model['strategy'] in ['greedy', 'e-greedy']:	
			msg2 = 'a=' + '{0:.2f}'.format(self.learning_rate)
			msg2 += ' d=' + '{0:.2f}'.format(self.discount_factor)

			if self.left_model['strategy'] == 'e-greedy':
				msg2 += ' e=' + '{0:.3f}'.format(self.epsilon)

			textsurface2 = self.font.render(msg2, False, WHITE)
			textrect2 = textsurface2.get_rect(center=(self.width / 2, self.height - MARGIN / 2))
			self.screen.blit(textsurface2, textrect2)

	def draw_game(self, phase):
		if not self.round_over:
			self.screen.fill(BLACK)
			self.static_draw()
			self.dynamic_draw(phase)
			pygame.display.update()
			self.clock.tick(FPS)

		else:
			color = RED if phase == 'train' else BLUE
			self.screen.fill(color)
			pygame.display.update()
			self.clock.tick(FPS_ROUND_OVER)

	################################## END GUI STUFF ##################################

# return curr game score
def play_game(args, game, phase='train'):
	
	global SLOW_MOTION, FPS, FPS_ROUND_OVER

	game.restart_game()
	game.restart_learning_module()

	while True:

		if game.left_curr_score > MAX_SCORE:
			print("MORE THAN MAX ITERATIONS")
			return game.left_curr_score

		for event in pygame.event.get():
				if event.type == pygame.QUIT:
					return game.left_curr_score

		# go to slow motion is SPACE is pressed
		pressed = pygame.key.get_pressed()
		if pressed[pygame.K_SPACE]:
			SLOW_MOTION = not SLOW_MOTION
			if SLOW_MOTION:
				FPS, FPS_ROUND_OVER = -1, -1
			else: 
				FPS, FPS_ROUND_OVER = 200, 5
		pygame.event.pump()

		game.move_ball()

		do_update = True if phase == 'train' else False

		left_action = game.get_action('left', do_update=do_update)
		game.apply_action(left_action, 'left', args.step)
		right_action = game.get_action('right', do_update=False)
		game.apply_action(right_action, 'right', args.step)

		
		game.draw_game(phase)
			
		if game.round_over:
			score = game.left_curr_score
			return score

		game.check_collision()
		game.check_ball_caught()	# after this we have the rewards
		game.update_ball_dir()

def main(args):
	game = pongGame(args)
	game.draw_game('train')

	train_data = []
	eval_data = []

	for train_episode in range(1, args.train_episodes + 1):
		score = play_game(args, game, phase='train')
		train_data.append(score)

		with open(args.output_file + '.txt', 'w') as f:
			for elem in train_data:
				f.write(str(elem) + ' ')
			f.close()
		

		if train_episode % args.eval_every == 0:
			avg_score = 0.0
			for eval_episode in range(1, args.eval_episodes + 1):
				curr_score = play_game(args, game, phase='eval')
				avg_score += curr_score
			avg_score /= args.eval_episodes
			eval_data.append(avg_score)

		if len(train_data) >= 5:
			train_data_plot = np.convolve(train_data, [0.2] * 5, 'same')
		else:
			train_data_plot = train_data

		# plot using the data we currently have
		plt.clf()
		plt.xlabel('Episode')
		plt.ylabel('Score')

		plt.plot(\
			np.linspace(1, train_episode, train_episode), \
			train_data_plot, \
			color='red', \
			linewidth=0.5)

		plt.plot(\
			np.linspace(args.eval_every, \
						train_episode - train_episode % args.eval_every, \
						len(eval_data)), \
			eval_data, \
			color='blue', \
			linewidth=2.0)

		plt.savefig(args.output_file + '.png')

def do_sanity_checks(args):
	try:
		assert (args.width - 2 * MARGIN) % args.step == 0
		assert (args.height - 2 * MARGIN) % args.step == 0
		assert ((args.width - 2 * MARGIN) / args.step) % 2 == 1
		assert ((args.height - 2 * MARGIN) / args.step) % 2 == 1
		assert (args.discount_factor >= 0)
		assert (args.discount_factor <= 1)
		assert (args.learning_rate >= 0)
		assert (args.learning_rate <= 1)
		assert (args.left_strategy in ['random', 'greedy', 'e-greedy'])
		assert (args.right_strategy in ['random', 'greedy', 'almost-perfect'])
		# assert (args.output_file is not None)
	except AssertionError as err:
		print("Argument error. Please check the arguments. Exiting.")
		sys.exit(-1)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	# grid parmeters
	parser.add_argument('--width', type=int, dest='width', default=410)
	parser.add_argument('--height', type=int,dest='height', default=310)
	parser.add_argument('--step', type=int, dest='step', default=20)

	# learning parameters
	parser.add_argument('--discount-factor', type=float, dest='discount_factor', default=0.5)
	parser.add_argument('--learning-rate', type=float, dest='learning_rate', default=0.98)
	parser.add_argument('--epsilon', type=float, dest='epsilon', default=0.05)
	
	# strategies
	parser.add_argument('--left-strategy', dest='left_strategy', default=None)
	parser.add_argument('--right-strategy', dest='right_strategy', default=None)
	
	# train / eval episodes
	parser.add_argument('--eval-every', type=int, dest='eval_every', default=10)
	parser.add_argument('--train-episodes', type=int, dest='train_episodes', default=500)
	parser.add_argument('--eval-episodes', type=int, dest='eval_episodes', default=10)
	
	parser.add_argument('--debug', dest='debug', default=True)
	parser.add_argument('--resume-left', dest='resume_left', default=None)
	parser.add_argument('--output-file', type=str, dest='output_file')

	args, unknown = parser.parse_known_args()
	
	do_sanity_checks(args)

	print(unknown)
	
	random.seed(3)
	main(args)
