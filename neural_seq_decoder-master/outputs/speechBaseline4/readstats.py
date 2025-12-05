#thanks piazza.

import pickle

with open('trainingStats', 'rb') as f:
    tStats = pickle.load(f)

print('testLoss: ', tStats['testLoss'])
print('testCER: ', tStats['testCER'])