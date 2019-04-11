import numpy as np
import random
import matplotlib.pyplot as plt

def checkWin(board, char):	

	for i in range(3):
		if board[i,0] == char and board[i,1] == char and board[i,2] == char:
			return True
	for i in range(3):
		if board[0,i] == char and board[1,i] == char and board[2,i] == char:
			return True

	if board[0,0] == char and board[1,1] == char and board[2,2] == char:			
		return True

	if board[0,2] == char and board[1,1] == char and board[2,0] == char:
		return True

	return False

def posActs(board):

	[i,j] = np.where(board=='_')
	acts = np.hstack((i.reshape(-1,1),j.reshape(-1,1)))

	return acts

def getID(state):

	return ''.join([str(val) for val in (state.flatten())])


def updateQ(Q, IDList, actList, reward, alpha, gamma):

	IDList.reverse()
	actList.reverse()
	for i,ID in enumerate(IDList):
		if i==0:
			if ID not in Q:
				Q[ID] = np.zeros((3,3),dtype=np.float32)
			Q[ID][actList[i][0],actList[i][1]] = reward
		else:
			if ID not in Q:
				Q[ID] = np.zeros((3,3),dtype=np.float32)
			lastRew = Q[ID][actList[i][0],actList[i][1]]
			bestFutRew = np.nanmax(Q[IDList[i-1]])
			Q[ID][actList[i][0],actList[i][1]] = (1-alpha)*lastRew+\
				alpha*(0+gamma*bestFutRew)
	return Q

def bestMove(ID, board, Q):

	maxVal = np.nanmax(Q[ID])
	indX,indY = np.where(Q[ID] == maxVal)
	ind = np.hstack((indX.reshape(-1,1),indY.reshape(-1,1)))
	randNum = random.choice(range(ind.shape[0]))
	#print(ind)
	move = (ind[randNum,0],ind[randNum,1])

	return move

def train(episodes, alpha1, gamma1, expFac, winX= False):

	Q1 = {}
	Q2 = {}
	if winX:
		expRat1 = 0.5
		expRat2 = 1.0
	else:
		expRat1 = 1.0
		expRat2 = 0.5
	win = 0
	lose = 0
	tie = 0
	invalid = 0
	for i in range(episodes):
		board = np.chararray((3,3))
		board[:] = '_'
		comp = False
		playSel = random.choice([1,2])
		stat = 0
		ID1List = []
		act1List = []
		ID2List = []
		act2List = []
		while not comp:

			
			if playSel == 1:
				sID1 = getID(board)
				ID1List.append(sID1) 
				acts1 = posActs(board)
				if random.random() < expRat1 or sID1 not in Q1:
					rowInd = random.choice(range(acts1.shape[0]))
					indAct1 = acts1[rowInd,:]
				else:
					indAct1 = bestMove(sID1, board, Q1)
				if board[indAct1[0],indAct1[1]] != '_':
					board[indAct1[0],indAct1[1]] = 'O'
					act1List.append((indAct1[0],indAct1[1]))
					Q1 = updateQ(Q1, ID1List, act1List, -10.0, alpha1, gamma1)
					if not winX:
						invalid+=1
						plt.subplot(2,2,4)
						plt.plot(i,invalid,'ob')
					comp = True
				board[indAct1[0],indAct1[1]] = 'O'
				act1List.append((indAct1[0],indAct1[1]))
				stat+=1
				

			if playSel == 2:
				sID2 = getID(board)
				ID2List.append(sID2)
				acts2 = posActs(board)
				if random.random() < expRat2 or sID2 not in Q2:
					rowInd = random.choice(range(acts2.shape[0]))
					indAct2 = acts2[rowInd,:]
				else:
					indAct2 = bestMove(sID2, board, Q2)
				if board[indAct2[0],indAct2[1]] != '_':
					board[indAct2[0],indAct2[1]] = 'X'
					act2List.append((indAct2[0],indAct2[1]))
					Q2 = updateQ(Q2, ID2List, act2List, -10.0, alpha1, gamma1)
					if winX:
						invalid+=1
						plt.subplot(2,2,4)
						plt.plot(i,invalid,'ob')
					comp = True
				board[indAct2[0],indAct2[1]] = 'X'
				act2List.append((indAct2[0],indAct2[1]))
				stat+=1

			if checkWin(board,'O'):
				Q1 = updateQ(Q1, ID1List, act1List, 1.0, alpha1, gamma1)
				Q2 = updateQ(Q2, ID2List, act2List, -1.0, alpha1, gamma1)
				if not winX:
					win+=1
					plt.subplot(2,2,1)
					plt.plot(i,win,'xg')
				if winX:
					lose+=1
					plt.subplot(2,2,2)
					plt.plot(i,lose,'xr')
				comp = True

			elif checkWin(board,'X'):
				Q1 = updateQ(Q1, ID1List, act1List, -1.0, alpha1, gamma1)
				Q2 = updateQ(Q2, ID2List, act2List, 1.0, alpha1, gamma1)
				if winX:
					win+=1
					plt.subplot(2,2,1)
					plt.plot(i,win,'xg')
				if not winX:
					lose+=1
					plt.subplot(2,2,2)
					plt.plot(i,lose,'xr')
				comp = True
				
			elif posActs(board).shape[0]==0:
				Q1 = updateQ(Q1, ID1List, act1List, 0.5, alpha1, gamma1)
				Q2 = updateQ(Q2, ID2List, act2List, 0.5, alpha1, gamma1)
				tie+=1
				plt.subplot(2,2,3)
				plt.plot(i,tie,'og')
				comp = True
				
			
			if playSel ==1:
				playSel = 2
			else:
				playSel = 1
		if winX:
			if expRat2 > 0.01:
				expRat2 *= expFac
		else:
			if expRat1 > 0.01:
				expRat1 *= expFac

	
	return Q1,Q2


def play(Q, human, bot):

	board = np.chararray((3,3))
	board[:] = '_'
	comp = False
	print("Enter 1 if you want to take chance first, else enter 2")
	turn = int(raw_input('-->'))
	if turn == 1:
		player = True
	else:
		player = False
	while not comp:

		if player:
			print("Enter the position of array")
			ip_pos = raw_input('-->')
			ip_pos = ip_pos.split(',')
			[i,j] = [int(ip_pos[0]),int(ip_pos[1])]
			board[i,j] = human
			print(board)
			if checkWin(board,human):
				print("You Win!!!")
				comp = True
			elif  posActs(board).shape[0] == 0:
				print("Its a tie!")
				comp = True


		if not player:
			sID = getID(board)
			acts1 = posActs(board)
			if sID not in Q:
				rowInd = random.choice(range(acts1.shape[0]))
				indAct = acts1[rowInd,:]
				print('state not trained')
			else:
				indAct = bestMove(sID, board, Q)
			board[indAct[0],indAct[1]] = bot
			print(board)
			if checkWin(board,bot):
				print("I Win!!!")
				comp = True
			elif  posActs(board).shape[0] == 0:
				print("Its a tie!")
				comp = True

		player = not player
	return