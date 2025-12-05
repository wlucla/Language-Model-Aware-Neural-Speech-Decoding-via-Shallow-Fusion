modelName = 'speechBaseline4'

args = {}
args['outputDir'] = '/home/jupyter/neural_seq_decoder-master/outputs/' + modelName
args['datasetPath'] = '/home/jupyter/neural_seq_decoder-master/competitionData/CD_Datasets.pkl'
args['seqLen'] = 150
args['maxTimeSeriesLen'] = 1200
args['batchSize'] = 128
args['lrStart'] = 0.05
args['lrEnd'] = 0.02
args['nUnits'] = 256
args['nBatch'] = 100000 #3000
args['nLayers'] = 5
args['seed'] = 0
args['nClasses'] = 40
args['nInputFeatures'] = 256
args['dropout'] = 0.2
args['whiteNoiseSD'] = 0.8
args['constantOffsetSD'] = 0.2
args['gaussianSmoothWidth'] = 2.0
args['strideLen'] = 4
args['kernelLen'] = 32
args['bidirectional'] = False
args['l2_decay'] = 1e-5


###My augmentations.
args["rollingZScored"] = 0      # Number of past neural signals to use. 0 means use no past.
# Probability of channel failure. Channels fail independently for an entire Trial. 
args["MaskFeaturesWithProbability"] = 0
#ADDING POISSON VARIATIONS IN SPIKE COUNTS
args['poisson?'] =False


###Anisha's Augmentations
args['time_masking_width']=0
args['time_feature_masking']=0
args['time_shift']=0
###



import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from neural_decoder.neural_decoder_trainer import trainModel

trainModel(args)


