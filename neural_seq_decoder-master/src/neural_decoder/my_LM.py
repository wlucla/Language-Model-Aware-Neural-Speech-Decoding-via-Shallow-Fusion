import math
import pickle
from collections import defaultdict, Counter
import re
#"""using the phoneme indexing in data processing"""
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
from collections import defaultdict, Counter
import re
class our_LM_n:
    #initialiser
    def __init__(self):
        #YOU MUST CREATEA DATA STRUCTURE THAT REMEMEBRS THE AMOUNT OF TIMES A PHONEME APPEARS AFTER ANOTHER 2 Other PHONEMEs (such is a 3 gram). THE LATTER IS THE 'KEY' OF THE DICT. THE FORMER IS A SIMPLE COUNTER THAT COUNTS OCCURENECES OF THE SUBSEQUENT PHONEMES. 
        self.ngrams= defaultdict(Counter)
        self.n=3 #this means ill try to predict the 3rd phoneme based on the previous 2 phoenemos
    def peruse_phoneme_dictionary(self, downloaded='USING_this_Dictionary.txt'):
        def remove_number(x):
            #remove weird numbers from phoneme dict
            return re.sub(r'\d', '', x)
        #I REALIZED THE TEXT FILE HAS SO MANY ';;;' BEGINNIGNGS, AND THOSE ARE COMMENTS. WASTED MORE THAN 3 HOURS CONFUSED ABOUT WHY MY CODE DIDNT WORK; SHOULDVE LOOKED AT THE FILE BEFOREHAND
        
        #make the class read the dictionary...
        with open(downloaded, 'r',encoding='utf-8') as _:
            for every_row in _:
                ###IMPORTANT***
                if every_row.startswith(';;;'):
                    continue
                    
                #form containers: lists for each row
                container =every_row.split()
                current_word= container[0]
                
                #i cannot deal with synonymous pronoucniations right now, reason is cuz idk how
                if ')' in current_word:
                    continue
                current_words_phonemes=container[1:]
                """CRITICAL ERROR-CMU dictionary contains NUMBERS IN THE PHONEMES, LUCKILY IF YOU REMOVE THEM IT MATCHES OUR TOTAL_VOCABULARY. REMOVE ALL\d DIGITS FROM THE VOCAB"""
                current_words_phonemes=[remove_number(_) for _ in current_words_phonemes]#stop the numbers in the phonemes
                current_words_phonemes=[re.sub(r'\d','',m) for m in current_words_phonemes]
                
                #notes
                #so if i had n=3 gram, i predict phoeneme from previous 2 pheomemes.
                #1,2,3,4,5-> 1,2=>3; 2,3=>4; 3,4=>5. I need 3 iterations. #phoenemes-n=2, 2+1=3
                
                for j in range(len(current_words_phonemes)-self.n+1):
                    #your 'prior' is the first n-1 pheonemes starting from j...use tuple so it can be keyed into our defaultdict intialization
                    prior=tuple(current_words_phonemes[j:j+self.n-1])
                    #intutitively, we hope use prior to predict the next phoenem...but over here we just store the next phoeneme from the dataset, no big deal
                
                    prediction=current_words_phonemes[j+self.n-1]
                    #key this pair into our default dict. The default dict tolerates COUNTERS, think of it as an unnormlaizsed probability 'distribution'
                    self.ngrams[prior][prediction]=self.ngrams[prior][prediction]+1 #COUNTER UPDATE.
                    
#"""based on our reading of the phoneme dictionary, how likely is an arbitrary string of phoneemes/READ THE IMPORTANT NOTE AT END OF FUNCTION: WE ARENT REALLY RETUNRING A PROBABILITY OF A WORD, MMORE OF A SCORING."""

    def probability_of_arbitrary_phonemeseq(self, arbitrary_phoneme_seq): #note, arbitrary_phoneme_seq should be consistent with the vraiable total_vocabulary: THIS IS EMPIRICAL. Dont include the BLank token.
        arbitrary_word_in_phoneme_representation=[total_vocabulary[_] for _ in arbitrary_phoneme_seq if _ !=0]
        if len(arbitrary_word_in_phoneme_representation)<self.n:
            return 0.00 # you must have at least 3 phonemes, or it doesnt make sense to predict the 3rd phoneme using the first 2.
        arbitrary_word_in_phoneme= ' '.join(arbitrary_word_in_phoneme_representation)
        if arbitrary_word_in_phoneme.strip()=='':
            return 0.00 #empty phoeneme representations of a word are unacceptable
        #as in homework 2, use log probabilities whenever possible since underflows are annoyingly hard to catch
        #number of keys in our model that has read the dictionary
        key_count = len(self.ngrams)
        #we'll be additive to this initialized probability
        ln_prob = 0.00
        
        #same loop loigic as above
        for k in range(len(arbitrary_word_in_phoneme_representation)-self.n+1):
            prior = tuple(arbitrary_word_in_phoneme_representation[k:k+self.n-1])
            predict =arbitrary_word_in_phoneme_representation[k+self.n-1]
            
            #count occurence of predict appearing after prior in our trained model, return 0 if predict doesnt appear ever
            occurences =self.ngrams[prior].get(predict,0)
            #all different phoenes that occur after prior...all instances
            every_occurence=sum(self.ngrams[prior].values())
            
           # """NOTE WE ARE NOT ATTEMPTING TO RETURN THE PROBABILITU OF THE WORD SINCE WE ARE MISSING THE STARTING-Phoneme PROBABILITIES, BUT STILL THIS SCORING IS SUFFICIENT TO DIFFERENTIATE LIKELY SEUQENCES OF PHONOMES FROM GIBBERISH."""
            temp_prob = 10**-8
            if every_occurence==0:
                temp_prob=10**-8
            else:
                temp_prob=max(10**-8, occurences/every_occurence)
            ln_prob= ln_prob+math.log(temp_prob)
        return ln_prob
            
   
#COMMENTED OUT BECAUSE CREATION OF PKL SUCCEEDED
# if __name__ == "__main__":           
#     #trianing/aka scoring
#     creation_of_model =our_LM_n()
#     creation_of_model.peruse_phoneme_dictionary('USING_this_Dictionary.txt')

#     #write to pickle
#     with open('OUR_TRAINED_LM.pkl', 'wb') as l:
#         pickle.dump(creation_of_model,l)
 
################################################          

# class our_LM_n:
#     def __init__(self):
        
#         self.ngrams= defaultdict(Counter)
#         self.n=3 
#     def peruse_phoneme_dictionary(self, downloaded='USING_this_Dictionary.txt'):
      
#         with open(downloaded, 'r',encoding='utf-8') as _:
#             for every_row in _:
               
#                 if every_row.startswith(';;;'):
#                     continue
                    
               
#                 container =every_row.split()
#                 current_word= container[0]
                
               
#                 if ')' in current_word:
#                     continue
#                 current_words_phonemes=container[1:]
                
                
#                 for j in range(len(current_words_phonemes)-self.n+1):
                   
#                     prior=tuple(current_words_phonemes[j:j+self.n-1])
                    
#                     prediction=current_words_phonemes[j+slef.n-1]
                   
#                     self.ngrams[prior][prediction]=self.ngrams[prior][prediction]+1 #COUNTER UPDATE.
    
#     def probability_of_arbitrary_phonemeseq(arbitrary_phoneme_seq, self): 
#         arbitrary_word_in_phoneme_representation=[total_vocabulary[_] for _ in arbitrary_phoneme_seq if _ !=0]
#         if len(arbitrary_word_in_phoneme_representation)<self.n:
#             return 0.00 
#         arbitrary_word_in_phoneme= ' '.join(arbitrary_word_in_phoneme_representation)
#         if arbitrary_word_in_phoneme.strip()=='':
#             return 0.00 
       
#         key_count = len(self.ngram)
       
#         ln_prob = 0.00
        
       
#         for k in range(len(arbitrary_word_in_phoneme_representation)-self.n+1):
#             prior = tuple(arbitrary_word_in_phoneme_representation[k:j+self.n-1])
#             predict =arbitrary_word_in_phoneme_representation[k+self.n-1]
            
           
#             occurences =self.ngram[prior].get(predict,0)
           
#             every_occurence=sum(self.ngrams[prior].values())
            
           
#             temp_prob = 10**-8
#             if every_occurence==0:
#                 temp_prob=10**-8
#             else:
#                 temp_prob=max(10**-8, occurences/every_occurence)
#             ln_prob= ln_prob+math.log(temp_prob)
            


# creation_of_model =our_LM_n()
# creation_of_model.peruse_phoneme_dictionary('USING_this_Dictionary.txt')
# with open('OUR_TRAINED_LM.pkl', 'wb') as l:
#     pickle.dump(creation_of_model,l)
        
        