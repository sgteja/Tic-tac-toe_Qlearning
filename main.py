import numpy as np 
from board import train,play
import pickle as pic
import argparse
import matplotlib.pyplot as plt


Parser = argparse.ArgumentParser()
Parser.add_argument('--Train', default='F',\
	help='boolean to train the model or not(Default is False)')
Args = Parser.parse_args()
t = Args.Train

if t == 'T':
	a1,_=train(episodes=10000, alpha1=0.4, gamma1=0.9, expFac=0.9)
	with open('tableO.pkl','wb') as f:
		pic.dump(a1, f, protocol = pic.HIGHEST_PROTOCOL)
	plt.show()
	_,a2=train(episodes=10000, alpha1=0.4, gamma1=0.9, expFac=0.9,winX=True)
	with open('tableX.pkl','wb') as f:
		pic.dump(a2, f, protocol = pic.HIGHEST_PROTOCOL)
	plt.show()
#print(a1)
else:
	print("Enter the character you want \'X\' or \'O\'")
	XO = (raw_input('-->'))
	if XO == 'X':
		with open('tableO.pkl','rb') as f:
			a1 = pic.load(f)
		play(a1,'X','O')
	else:
		with open('tableX.pkl','rb') as f:
			a1 = pic.load(f)
		play(a1,'O','X')
	