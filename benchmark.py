import atari_py
import subprocess
from joblib import Parallel, delayed
import os
import time
from collections import defaultdict

MAX_RETRY = 3
failures = defaultdict(int)
games = []
gpus = None
rapid = False
idx = 0
	
benchmark_id = 'v1'

def get_run_id(benchmark_id, options, game):
	method = "_".join(['rapid' if rapid else 'standard'] + options)
	run_id = 'benchmark_{}/{}/{}'.format(benchmark_id, method, game)
	run_id = run_id.replace('_--test-ensemble', '')
	return run_id

def get_score(run_id):
	rewards_path = os.path.join('results', run_id, 'rewards.tsv')
	# print(rewards_path)
	if os.path.exists(rewards_path):
		with open(rewards_path, 'r') as f:
			data = f.readlines()
		score = None
		for line in data:
			if '100000' in line:
				score = float(line.split('\t')[-1]) # get the latest
	else:
		score = None
	return score

def get_ensemble_score(run_id, tid):
	rewards_path = os.path.join('results', run_id, 'ensemble_rewards.tsv')
	# print(rewards_path)
	if os.path.exists(rewards_path):
		with open(rewards_path, 'r') as f:
			data = f.readlines()
		score = None
		for line in data:
			if tid in line:
				score = float(line.split('\t')[-1]) # get the latest
	else:
		score = None
	return score

def train(options):
	global idx
	if len(games) == 0:
		return
	game = games.pop(0)
	run_id = get_run_id(benchmark_id, options, game)

	test_ensemble = '--test-ensemble' in options
	if not test_ensemble and get_score(run_id) is not None: # skip if done
		return

	if get_ensemble_score(run_id, 'tid-mean') is not None: # skip if done
		return

	gpu = gpus.pop(0)
	print('[START] id: {}, game: {}, GPU: {}'.format(idx, game, gpu))
	program = ['./rapid.sh'] if rapid else ['python', 'main.py']
	print('==> Run ID:', run_id)
	subprocess.run(program + ['--game', game, '--id', run_id] + options, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)))
	gpus.append(gpu)

	if get_score(run_id) is None: # retry later on failure
		if failures[game] >= MAX_RETRY:
			return
		failures[game] += 1
		print('[FAILED] id: {}, game: {}, GPU: {}'.format(idx, game, gpu))
		games.append(game)
		return

	idx += 1
	print('[END] id: {}, game: {}, GPU: {}'.format(idx, game, gpu))

if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser(description='Benchmark')
	parser.add_argument('--gpus', nargs='+', type=int)
	parser.add_argument('--rapid', action='store_true')
	parser.add_argument('--all', action='store_true')
	parser.add_argument('--options', type=str, default="")
	parser.add_argument('--benchmark_id', type=str, default=benchmark_id)
	parser.add_argument('--games', nargs='+', type=str)
	parser.add_argument('--result', action='store_true')
	parser.add_argument('--ensemble-result', action='store_true')

	args = parser.parse_args()
	print(args)

	gpus = args.gpus
	rapid = args.rapid
	options = args.options.split(" ") if len(args.options) > 0 else []
	benchmark_id = args.benchmark_id
	idx = 0

	game_list = atari_py.list_games()
	games = ['alien', 'amidar', 'assault', 'asterix', 'bank_heist', 'battle_zone', 'boxing', 'breakout', 'chopper_command', 
			'crazy_climber', 'demon_attack', 'freeway', 'frostbite', 'gopher', 'hero', 'jamesbond', 'kangaroo', 'krull', 
			'kung_fu_master', 'ms_pacman', 'pong', 'private_eye', 'qbert', 'road_runner', 'seaquest', 'up_n_down']

	if args.games is not None and len(args.games) > 0:
		games = args.games

	for game in games:
		assert game in game_list

	if args.all:
		games = game_list

	if args.result:
		for game in games:
			run_id = get_run_id(benchmark_id, options, game)
			score = get_score(run_id)
			if score is None:
				print()
			else:
				print(score)
		exit()

	if args.ensemble_result:
		for i in list(range(5))+['mean']:
			tid = f'tid-{str(i)}'
			print(tid)
			print('-'*len(tid))
			for game in games:
				run_id = get_run_id(benchmark_id, options, game)
				score = get_ensemble_score(run_id, tid)
				if score is None:
					print()
				else:
					print(score)
			print()
		exit()

	n_upperbound = len(games) * MAX_RETRY

	Parallel(n_jobs=len(gpus), require='sharedmem')(delayed(train)(options) for _ in range(n_upperbound))
