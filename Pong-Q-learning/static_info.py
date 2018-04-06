MINUS_REWARD = -1
PLUS_REWARD = 1
NO_REWARD = 0

MARGIN = 15
PAD_WIDTH = 5
PAD_OFFSET = MARGIN - PAD_WIDTH
PAD_HEIGHT = 15
BALL_SIZE = 7

BLACK     = (0  ,0  ,0  )
WHITE     = (255,255,255)
BLUE      = (0, 0, 255)
GRAY      = (125, 125, 125)
RED       = (255, 0, 0)

BALL_DIRS = [(1, 1), (-1, -1), (1, -1), (-1, 1)]
ACTIONS = ['up', 'down', 'nop']
NUM_ACTIONS = len(ACTIONS)
NUM_BALL_DIRS = len(BALL_DIRS)

INF = -int(1e15)
ITERATIONS = 2000
K = 0.001

EPS = 0.1	# used for 'almost-perfect' opponent strategy

MAX_SCORE = 100

SLOW_MOTION = False
if not SLOW_MOTION:
	FPS = -1
	FPS_ROUND_OVER = -1
else:
	FPS = 150
	FPS_ROUND_OVER = 1

DEBUG = 0
