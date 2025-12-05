from .augmentations import Rolling_Z_Scores 
from .augmentations import adding_poison_noise
from .augmentations import Feature_masking
from .augmentations import time_masking
from .augmentations import time_feature_masking
from .augmentations import time_shift


import os
import pickle
import time

from edit_distance import SequenceMatcher
import hydra
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from .model import GRUDecoder
from .dataset import SpeechDataset

#USe the phoneme indexing provided by the data Formatting or else.
import math
import pickle
from collections import defaultdict, Counter
import re
total_vocabulary={
    0: '', 
    1: 'AA', 
    2: 'AE',
    3: 'AH',
    4: 'AO',
    5: 'AW',  
    6: 'AY',
    7: 'B',
    8: 'CH',
    9: 'D', 
    10: 'DH' ,
    11: 'EH',
    12: 'ER',
    13: 'EY',
    14: 'F',
    15: 'G',
    16: 'HH',
    17: 'IH',
    18: 'IY',
    19: 'JH',
    20: 'K',   
    21: 'L',
    22: 'M',
    23: 'N',
    24: 'NG',
    25: 'OW',
    26: 'OY',
    27: 'P',
    28: 'R',
    29: 'S',
    30: 'SH',
    31: 'T',
    32: 'TH',
    33: 'UH',
    34: 'UW',
    35: 'V',
    36: 'W',
    37: 'Y',
    38: 'Z',
    39: 'ZH',
    40: 'SIL'
}
#copy and pasting from my_LM, nothing new. I did it because i cant figure that paths, and execution in terminal always fails. I removed the comments here for clarity. Refer to my_LM.py for comments
class our_LM_n:
    def __init__(self):
        self.ngrams = defaultdict(Counter)
        self.n = 3
    def peruse_phoneme_dictionary(self, downloaded='/home/jupyter/neural_seq_decoder-master/src/neural_decoder/USING_this_Dictionary.txt'):
        def remove_number(x):
            #edit: remove weird numbers from phoneme dict
            return re.sub(r'\d', '', x)
        with open(downloaded, 'r', encoding='utf-8') as _:
            for every_row in _:
                if every_row.startswith(';;;'):
                    continue
                container = every_row.split()
                current_word = container[0]
                if ')' in current_word:
                    continue
                current_words_phonemes = container[1:]
                current_words_phonemes=[remove_number(_) for _ in current_words_phonemes]
                for j in range(len(current_words_phonemes) - self.n + 1):
                    prior = tuple(current_words_phonemes[j:j + self.n - 1])
                    prediction = current_words_phonemes[j + self.n - 1]
                    self.ngrams[prior][prediction] = self.ngrams[prior][prediction] + 1
                    
    def probability_of_arbitrary_phonemeseq(self, arbitrary_phoneme_seq):
        arbitrary_word_in_phoneme_representation = [total_vocabulary[_] for _ in arbitrary_phoneme_seq if _ != 0]
        if len(arbitrary_word_in_phoneme_representation) < self.n:
            return 0.00 
        arbitrary_word_in_phoneme = ' '.join(arbitrary_word_in_phoneme_representation)
        if arbitrary_word_in_phoneme.strip() == '':
            return 0.00 
        key_count = len(self.ngrams)       
        ln_prob = 0.00
      
        for k in range(len(arbitrary_word_in_phoneme_representation) - self.n + 1):
            prior = tuple(arbitrary_word_in_phoneme_representation[k:k + self.n - 1])
            predict = arbitrary_word_in_phoneme_representation[k + self.n - 1]
            occurences = self.ngrams[prior].get(predict, 0)
            every_occurence = sum(self.ngrams[prior].values())
            temp_prob = 10**-8
            if every_occurence == 0:
                temp_prob = 10**-8
            else:
                temp_prob = max(10**-8, occurences / every_occurence)
            ln_prob = ln_prob + math.log(temp_prob)
        return ln_prob
    
#VERY VOLATILE> NOT SURE IF THIS WILL WORK. what if you have an internal AUTOCORRECTOR that tries to force decodedSeq to behave more like the english language?
def autocorrect(decodedSeq, model, iterate=1):
    sequence_to_return=decodedSeq.copy() #improve upon this sequence...?
    current_score = model.probability_of_arbitrary_phonemeseq(decodedSeq)
    for i in range(iterate):
        for k in range(len(decodedSeq)):
            for j in range(1,41):
                test_temp_seq=sequence_to_return.copy()
                test_temp_seq[k]=j
                temp_score = model.probability_of_arbitrary_phonemeseq(test_temp_seq)
                if temp_score>current_score:
                    sequence_to_return=test_temp_seq
                    current_score=temp_score
    return sequence_to_return

#CREATE NEW LM INSTANCE AND TRAIN IT ON CMU DICTIONARY
my_lm = our_LM_n()
my_lm.peruse_phoneme_dictionary('/home/jupyter/neural_seq_decoder-master/src/neural_decoder/USING_this_Dictionary.txt')


from scipy.special import logsumexp #NOT WORKING FOR SOME REASON.

#I am not confiendent nor sure about how to play around with beam searches, so i will keep our methods to be greedy decoding, except ill let the LM scores obtained from my our_LM_n class influence the greeding decoding a bit.


def modded_greed(logit_scores_at_layers, our_model, alpha=0.05): #alpha,as a hyperparameteer, will weight the imporance of LM scores
    logit_scores_at_layers= logit_scores_at_layers.cpu().numpy()
    final_decode=[]
    for o in range(len(logit_scores_at_layers)): #iterate through time
        #take the top 3 ctc-scored logits at every time point. You are getting the vocabulary numbers
        best_logits_phone=[]
        top_3 =np.argsort(logit_scores_at_layers[o])[-3:][::-1]
        for p in top_3:
            if p==0:
                continue
            else:
                best_logits_phone.append(p)
        if len(best_logits_phone)==0:
            continue #nothing left to do.
        
        #make sure CTC scores are in log(softmax prob)
        lnCTC_score = logit_scores_at_layers[o]-logsumexp(logit_scores_at_layers[o]) #log of division is subtraction
        #best phoneme at a time point according to our LM scoring
        top_score =-float('inf') #initialize to very negative
        top_phoneme= best_logits_phone[0] #just take the best CTC phoeneme as default
        
        #perhaps this is a design flaw, but realize that the first n-1 phonemes only use CTC. There isnt neough info in final_decode yet to influence their scoring yet
        for q in best_logits_phone:
            #what to test...YOU MUST USE YOUR FINAL_DECODE info here
            temp_test_seq= final_decode+[q]
            
            #use our own scorig function
            temp_LM_SCORE=my_lm.probability_of_arbitrary_phonemeseq(temp_test_seq)
            
            relevant_CTC_lnProb=lnCTC_score[q]
            
            #from lecture notes, the shallowfusion score
            total_score= (1-alpha)*relevant_CTC_lnProb+alpha*temp_LM_SCORE
            
            if total_score> top_score:
                top_score=total_score
                top_phoneme=q
            
        #honor the ctc collapse rule.
        if len(final_decode)==0 or final_decode[-1]!= top_phoneme:
            final_decode.append(top_phoneme)
    return np.array(final_decode)
            
def getDatasetLoaders(
    datasetName,
    batchSize,
):
    with open(datasetName, "rb") as handle:
        loadedData = pickle.load(handle)

    def _padding(batch):
        X, y, X_lens, y_lens, days = zip(*batch)
        X_padded = pad_sequence(X, batch_first=True, padding_value=0)
        y_padded = pad_sequence(y, batch_first=True, padding_value=0)

        return (
            X_padded,
            y_padded,
            torch.stack(X_lens),
            torch.stack(y_lens),
            torch.stack(days),
        )

    train_ds = SpeechDataset(loadedData["train"], transform=None)
    test_ds = SpeechDataset(loadedData["test"])

    train_loader = DataLoader(
        train_ds,
        batch_size=batchSize,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=_padding,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batchSize,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=_padding,
    )

    return train_loader, test_loader, loadedData

def trainModel(args):
    os.makedirs(args["outputDir"], exist_ok=True)
    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])
    device = "cuda"

    with open(args["outputDir"] + "/args", "wb") as file:
        pickle.dump(args, file)

    trainLoader, testLoader, loadedData = getDatasetLoaders(
        args["datasetPath"],
        args["batchSize"],
    )

    model = GRUDecoder(
        neural_dim=args["nInputFeatures"],
        n_classes=args["nClasses"],
        hidden_dim=args["nUnits"],
        layer_dim=args["nLayers"],
        nDays=len(loadedData["train"]),
        dropout=args["dropout"],
        device=device,
        strideLen=args["strideLen"],
        kernelLen=args["kernelLen"],
        gaussianSmoothWidth=args["gaussianSmoothWidth"],
        bidirectional=args["bidirectional"],
    ).to(device)

    loss_ctc = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args["lrStart"],
        betas=(0.9, 0.999),
        eps=0.1,
        weight_decay=args["l2_decay"],
    )
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=args["lrEnd"] / args["lrStart"],
        total_iters=args["nBatch"],
    )
    
    #i feel like the model has the potential to learn more but gets interrupted when LR decreases too much. Ill test this schedular to see whats up
    # scheduler = torch.optim.lr_scheduler.LinearLR(
    #     optimizer,
    #     start_factor=1.0,
    #     end_factor=0.5,
    #     total_iters=args["nBatch"]*2,
    # )

    # --train--
    testLoss = []
    testCER = []
    startTime = time.time()
    for batch in range(args["nBatch"]):
        model.train()

        X, y, X_len, y_len, dayIdx = next(iter(trainLoader))
        X, y, X_len, y_len, dayIdx = (
            X.to(device),
            y.to(device),
            X_len.to(device),
            y_len.to(device),
            dayIdx.to(device),
        )

        # Noise augmentation is faster on GPU
        if args["whiteNoiseSD"] > 0:
            X += torch.randn(X.shape, device=device) * args["whiteNoiseSD"]

        if args["constantOffsetSD"] > 0:
            X += (
                torch.randn([X.shape[0], 1, X.shape[2]], device=device)
                * args["constantOffsetSD"]
            )
            
#my augmentations
        if args['poisson?']==True:
#add poisson variations to spike counts. 
            X = adding_poison_noise(X)
            
#turn on model the of faulty electrodes.
        if args["MaskFeaturesWithProbability"]>0:
            X= Feature_masking(X, electrode_failure_probability = args["MaskFeaturesWithProbability"])
            
            
            
#anisha's augmentations
        if args['time_masking_width']>0:
            X=time_masking(X, args['time_masking_width'])
            
        if args['time_feature_masking']>0:
            X=time_feature_masking(X, args['time_feature_masking'])
            
        if args['time_shift']>0:
            X=time_shift(X, args['time_shift'])
        

        
#turn on my rolling z score nnow, after all base augmetnations.
        if args["rollingZScored"]>0:
            X = Rolling_Z_Scores(X, desired_window=args["rollingZScored"])
            
            
        #Compute prediction error
        pred = model.forward(X, dayIdx)

        loss = loss_ctc(
            torch.permute(pred.log_softmax(2), [1, 0, 2]),
            y,
            ((X_len - model.kernelLen) / model.strideLen).to(torch.int32),
            y_len,
        )
        loss = torch.sum(loss)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # print(endTime - startTime)

        # Eval
        if batch % 100 == 0:
            with torch.no_grad():
                model.eval()
                allLoss = []
                total_edit_distance = 0
                total_seq_length = 0
                for X, y, X_len, y_len, testDayIdx in testLoader:
                    X, y, X_len, y_len, testDayIdx = (
                        X.to(device),
                        y.to(device),
                        X_len.to(device),
                        y_len.to(device),
                        testDayIdx.to(device),
                    )

                    pred = model.forward(X, testDayIdx)
                    loss = loss_ctc(
                        torch.permute(pred.log_softmax(2), [1, 0, 2]),
                        y,
                        ((X_len - model.kernelLen) / model.strideLen).to(torch.int32),
                        y_len,
                    )
                    loss = torch.sum(loss)
                    allLoss.append(loss.cpu().detach().numpy())

                    adjustedLens = ((X_len - model.kernelLen) / model.strideLen).to(
                        torch.int32
                    )
                    for iterIdx in range(pred.shape[0]):
                        #baseline greedy decoding
                        decodedSeq = torch.argmax(
                            torch.tensor(pred[iterIdx, 0 : adjustedLens[iterIdx], :]),
                            dim=-1,
                        )  # [num_seq,]
                        decodedSeq = torch.unique_consecutive(decodedSeq, dim=-1)
                        decodedSeq = decodedSeq.cpu().detach().numpy()
                        decodedSeq = np.array([i for i in decodedSeq if i != 0])
                        
                        # #LM shallow fuse
                        # # lm = my_lm.probability_of_arbitrary_phonemeseq(decodedSeq)
                        # #PRINT A FEW SCORES TO VERIFY SCORING IS RUNNING NROMALLY
                        # # print(f'LM Score: {lm}')
                        # # print(f'{decodedSeq}')
                        # #TAKES INTO CONSIDERATION THE LM SCORE AND THE CTC SCORE.
                        # OUT_logit =pred[iterIdx, 0 : adjustedLens[iterIdx], :]
                        # # print(OUT_logit)
                        # decodedSeq=modded_greed(OUT_logit,my_lm, alpha=0.2)
                        #autocorrect takes crazy computation, maybe in the future with more gpus i can do it.
                        # decodedSeq= autocorrect(decodedSeq, my_lm, 1)
                        trueSeq = np.array(
                            y[iterIdx][0 : y_len[iterIdx]].cpu().detach()
                        )

                        matcher = SequenceMatcher(
                            a=trueSeq.tolist(), b=decodedSeq.tolist()
                        )
                        total_edit_distance += matcher.distance()
                        total_seq_length += len(trueSeq)

                avgDayLoss = np.sum(allLoss) / len(testLoader)
                cer = total_edit_distance / total_seq_length

                endTime = time.time()
                print(
                    f"batch {batch}, ctc loss: {avgDayLoss:>7f}, cer: {cer:>7f}, time/batch: {(endTime - startTime)/100:>7.3f}"
                )
                startTime = time.time()

            if len(testCER) > 0 and cer < np.min(testCER):
                torch.save(model.state_dict(), args["outputDir"] + "/modelWeights")
            testLoss.append(avgDayLoss)
            testCER.append(cer)

            tStats = {}
            tStats["testLoss"] = np.array(testLoss)
            tStats["testCER"] = np.array(testCER)

            with open(args["outputDir"] + "/trainingStats", "wb") as file:
                pickle.dump(tStats, file)


def loadModel(modelDir, nInputLayers=24, device="cuda"):
    modelWeightPath = modelDir + "/modelWeights"
    with open(modelDir + "/args", "rb") as handle:
        args = pickle.load(handle)

    model = GRUDecoder(
        neural_dim=args["nInputFeatures"],
        n_classes=args["nClasses"],
        hidden_dim=args["nUnits"],
        layer_dim=args["nLayers"],
        nDays=nInputLayers,
        dropout=args["dropout"],
        device=device,
        strideLen=args["strideLen"],
        kernelLen=args["kernelLen"],
        gaussianSmoothWidth=args["gaussianSmoothWidth"],
        bidirectional=args["bidirectional"],
    ).to(device)

    model.load_state_dict(torch.load(modelWeightPath, map_location=device))
    return model


@hydra.main(version_base="1.1", config_path="conf", config_name="config")
def main(cfg):
    cfg.outputDir = os.getcwd()
    trainModel(cfg)

if __name__ == "__main__":
    main()