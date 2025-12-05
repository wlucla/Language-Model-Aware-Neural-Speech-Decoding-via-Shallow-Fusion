import math
import numbers
import torch
from torch import nn
from torch.nn import functional as F
import numpy


class WhiteNoise(nn.Module):
    def __init__(self, std=0.1):
        super().__init__()
        self.std = std

    def forward(self, x):
        noise = torch.randn_like(x) * self.std
        return x + noise

class MeanDriftNoise(nn.Module):
    def __init__(self, std=0.1):
        super().__init__()
        self.std = std

    def forward(self, x):
        _, C = x.shape
        noise = torch.randn(1, C) * self.std
        return x + noise

class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [torch.arange(size, dtype=torch.float32) for size in kernel_size]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= (
                1
                / (std * math.sqrt(2 * math.pi))
                * torch.exp(-(((mgrid - mean) / std) ** 2) / 2)
            )

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer("weight", kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                "Only 1, 2 and 3 dimensions are supported. Received {}.".format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups, padding="same")

    
    


""" notes from my data Augmentation Implementations"""

# def Rolling_Z_Scores(X, desired_window=0):
#     #X.shape= [batch, time, features]
#     batch_count, time_duration, feature_size = X.shape

#     #X is a batch, so you need to to a normalization for each trail or the logic is weird
#     #get everything you need for emprical variance.
#     #cumulative sums make things run faster when you want to do for exampe rooling means.
#     #[x0, x0+x1, x0+x1+x2, ...] mean(i=1, i=2) = (x0+x1+x2)-(x0) / (2-1+1): easier
#     sum_X = torch.cumsum(X, dim=1)
#     sum_X_sqr = torch.cumsum(X ** 2, dim=1)


#     #now you cna do rolling stuff, figure out exactly how you roll. use windows.
#     #note: for each time point, your window could be not-filled if at starts of trials

#     #logic notes
#     #t=0 got  no causal past.should get 1 for window length
#     #t=1 got 1 causal past. should get 2 for window length
#     #t=2 got 2 causal past. should get 3 for window length
#     #......SHould contiue up until causal_past_count
#     #t=causal_past_count got causal_past_count causal past. should get causal_past_count+1 for window length
#     #ok i think


#     #where does window lie at time t?
#     #negative start indices got clamp->0
#     #https://stackoverflow.com/questions/74183588/why-is-torch-clamp-in-pytorch-not-working

#     window_initial = torch.clamp(torch.arange(time_duration, device=X.device) - desired_window+1, min=0)  #shape [time,]
#     #trial length got padded, dont worry about dimensions.

#     #window_initial is the vector of start indices for each time point.

#     #get cumulative sums at the start.
#     sum_start =sum_X[:, window_initial, :] #amount accumuated before rollig window starts. this is the 'x0' we subtracted earlier int he example comment
#     sum_sqr_start = sum_X_sqr[:, window_initial, :]

#     ###NOTE I JUST REALIZED THAT I SHOULD NOT BE INCLUDINF THE START TIME POINT IN THE CUMULATIVE SUM I AM GOING TO BE SUBTRACTING. THAT WOULD MESS UP THE SUM OVER THE WINDOW OF INTEREST. i guess it doesnt really matter but ill fix later,

#     ##elements in window at time t
#     elements_in_window = torch.arange(time_duration, device=X.device) - window_initial + 1
#     ####personal data structure derivation: think about window length =2
#     #torch,arange(5) = [0,1,2,3,4]
#     #subtract the windo_initial [0,0,1,2,3] = [0,1,1,1,1]
#     #add 1 = [1,2,2,2,2] is that  not the number of elements in the window at each time point?
#     #dividing by tensors later, make sure refit to to tensor
#     elements_in_window =elements_in_window.view(1, time_duration, 1)


#     #now get the rolling sums
#     #feautre(start)+;;;+feature(t)
#     #using the window_initial, figure out the index right before the intial indices
#     before_start = (window_initial-1).clamp(min=0)
#     #for example a rolling window of size 2
#     #window_initial = [0,0,1,2,3]
#     #-1-> [-1, -1, 0,1,2]
#     #clamp [0, 0, 0, 1,2] #\INDICES RIGHT BEFORE INITIAL INDIDCES

#     previousCUMSUM= sum_X[:, before_start, :]

#     #this is exactly this example in code: [x0, x0+x1, x0+x1+x2, ...] mean(i=1, i=2) = (x0+x1+x2)-(x0) / (2-1+1):
#     rolling_sum = sum_X-previousCUMSUM #STILL batch*time*features in shape but contains rolling sums at al times now
#     previousCUMSUM_sqr = sum_X_sqr[:, before_start,:]
#     rolling_sum_sqr = sum_X_sqr-previousCUMSUM_sqr

#     roll_empirical_mean = rolling_sum/elements_in_window
#     roll_empircal_varaince = rolling_sum_sqr/elements_in_window-(roll_empirical_mean**2)
#     roll_empirical_std = torch.sqrt(roll_empircal_varaince)
#     #ready to rescale entire batch.
#     rolled_batch = (X-roll_empirical_mean) / roll_empirical_std


#THIS IS JUST A POLISHED VERSION OF THE DRAFT ABOVE for ROLLINGZSCORE: i do not want to erase the draft because the data structure is very obscure an i could forget my logic after a while if want to debug
# def Rolling_Z_Scores(X, desired_window=0):

#     batch_count, time_duration, feature_size = X.shape
    
#     device = X.device #why it keep saying i have the cuda cpu mismatch error? i literally put this here. edit: its working again and idk why; too scared to remove this line. its staying here.
    
#     sum_X = torch.cumsum(X, dim=1)
#     sum_X_sqr = torch.cumsum(X ** 2, dim=1)
#     window_initial = torch.clamp(torch.arange(time_duration, device=X.device) - desired_window+1, min=0)
#     elements_in_window = torch.arange(time_duration, device=X.device) - window_initial + 1

#     elements_in_window =elements_in_window.view(1, time_duration, 1)


#     before_start = (window_initial-1).clamp(min=0)
#     previousCUMSUM= sum_X[:, before_start, :]
#     rolling_sum = sum_X-previousCUMSUM +X[:, window_initial, :]
#     previousCUMSUM_sqr = sum_X_sqr[:, before_start,:]
#     rolling_sum_sqr = sum_X_sqr-previousCUMSUM_sqr+X[:, window_initial, :]**2

#     roll_empirical_mean = rolling_sum/elements_in_window
#     roll_empircal_varaince = rolling_sum_sqr/elements_in_window-(roll_empirical_mean**2)
#     roll_empircal_varaince = torch.clamp(roll_empircal_varaince, min=10**-4)
#     roll_empirical_std = torch.sqrt(roll_empircal_varaince)

#     rolled_batch = (X-roll_empirical_mean) / roll_empirical_std
#     return rolled_batch


#PROBLEM> WHY ARE YOU ROLL Z SCORRING THE SPIKE BANDPOWER TOGETHER WITH TXings????? IT SHOULD BE SEPARATE.
# def Rolling_Z_Scores(X, desired_window=0):
#     batch_count, time_duration, feature_size = X.shape
    
#     if desired_window==0:
#         return X
    
#     #what gets returned must be the same shape as X
#     rolls_Z = torch.zeros_like(X)
    
#     #iterate throgh the trials within one batch
#     for trial in range(batch_count):
#         #iter through the times within on trial
#         for time in range(time_duration):
#             #your start index for rolling should get constrained to time 0
#             start_index = max(0, time - desired_window+1) 
#             #obtain the end index,causually and exclusivley at the time+1 index
#             end_index =time+1
            
#             #extract the relevant features from the current trial
#             relevant_features =X[trial, start_index:end_index,:]
            
#             #calculate the mean in this window. 
#             relevant_features_mean=torch.mean(relevant_features, dim=0)
#             #compute empiricla std. with sample statisitics
#             relevant_features_std=torch.std(relevant_features, unbiased=False, dim=0)
            
#             #normalizations when dividing by the std: prevent division by zeros std.
#             relevant_features_std=torch.clamp(relevant_features_std, min=10**-4)
            
#             #standardize step.
#             rolls_Z[trial,time,:]=(X[trial,time,:]-relevant_features_mean)/relevant_features_std
        
#     return rolls_Z

#FIXED VERSION BELOW>FOR ROLLING Z SCORE. i just used forloops and stopped the cumulative sum logic because its giving me a stroke
def Rolling_Z_Scores(X, desired_window=0):
    batch_count, time_duration, feature_size = X.shape
    
    if desired_window==0:
        return X
    
    device=X.device

    rolls_Z = torch.zeros_like(X)

    for trial in range(batch_count):

        for time in range(time_duration):

            start_index = max(0, time - desired_window+1) 

            end_index =time+1
            

            relevant_features =X[trial, start_index:end_index,:]
            
            #ROLLING ZSCORE SHOULD BE APPLIED SEPARAETELY TO TX AND THE SBP. ELse the logic is identical to my previous incorrect implementation.
            
            #0-127 are TXs
            txings= relevant_features[:,:128]
            tx_mean=torch.mean(txings,dim=0)
            tx_std=torch.std(txings,unbiased=False,dim=0)
            tx_std=torch.clamp(tx_std, min=10**-4)
            
            #128-: are spbs
            spb= relevant_features[:,128:]
            spb_mean=torch.mean(spb, dim=0)
            spb_std=torch.std(spb,unbiased=False,dim=0)
            spb_std=torch.clamp(spb_std, min=10**-4)
            

            #separately do the z-scoring for tx and spb
            rolls_Z[trial, time, :128]=(X[trial, time,:128]-tx_mean)/tx_std
            rolls_Z[trial, time, 128:]=(X[trial, time, 128:]-spb_mean)/spb_std
        
    return rolls_Z
            
    


#randomly zero out entire channel's features in one trial (all time) within a batch. I believe this might happen in real life: failure of one channel doesnt necessarily mean the surrounding channels fail.

#NOTE in dataFormatting, the tx and sbp were CONCATENATED. So for a feature vector, to zero out electrode 0 means to set index 0 and index 128 of a feature vector to 0

def Feature_masking(X, electrode_failure_probability):
    #get the dimesnion of the batch known as X...
    batch_count, trials, features = X.shape
    
    device=X.device
    
    #create a matrix 1x128, filled with 0-1 values, we will apply a boolean filter to make it  binary row-vector thingy after which i'll use to zero out entire electrodes. Can think of it as multiplying this to a trial matrix to zero out features.
    dead_electrodes = torch.rand(batch_count, 128, device = X.device)<electrode_failure_probability
    
    #generate a binary bask container, will intialize to trues.
    BOOLmask = torch.ones(batch_count, 256, device = X.device)
    
    #YOU HAVE TO ZERO OUT ENTIRE ELECTRODES, not single features of an electrode or that does make sense in the real world
    
    #index into the dead electrodes in the batch and zero them all out.
    BOOLmask[:, 0:128][dead_electrodes]=0
    BOOLmask[:, 128:256][dead_electrodes]=0
    
    #you need to have trials count of the boolean mask, then ill multiply the mask to the Xth batch
    BOOLmask = BOOLmask.unsqueeze(1) #must be same dimensions as X
    
    return X*BOOLmask#filter the dead electrodes out. in the lectur
 
#what if for each training batch X, we augemnt each X[batch, trial, : (all features that are SPIKE COUNTS, NOT SPB)] with some poisson noise? Since each channel will record spike counts and we know spike count to be poisson distributed? For example, what if I take the current spike counts as the mean and then randomly repace the spike count with a randomly drawn integer out of the Poisson(current spike count) distiribution?
#i realized that this function doent really make snese...neuron activity is already posioon processes so probably have this noise already...why am i again laying poisson type noise on top? --include this in the writeup
import random
def adding_poison_noise(X, p=0.1):
    temp=random.uniform(0,1) #if p<temp, we will apply noise
    if p>temp:
        #use same logical style as above.

        batch_count, trials, features = X.shape

        device=X.device
        #get/alternate only the features in every batch+every trial that represent TX-ings
        TXing = X[:, :, 0:128]
    

        PoissonedTXing = torch.poisson(torch.clamp(TXing, min=0)) #let the data vary as a poisson process would so the model learns  the poisson not the memorizing the data. Im really confused why compiler complains that my TXing values are negative; should TXings be [0, inf)? i fixed it by adding a clamp but i dont understand why

        toreturn = torch.cat([PoissonedTXing, X[:, :, 128:]], dim=-1)#dim =-1 to make the features 256D againl. 
    
    
        return toreturn
    else: #do nothing.
        return X
 
 

####Anisha's augmentations.
def time_masking(X, max_width, p=0.10):
    
#In real neural recordings, there can be short periods where data is lost and this
#augmentation simulates those brief intervals by zeroing out a random stretch
#of time across all features for each trial
    

    #with probability (1 - p), skip augmentation and return X unchanged
    if torch.rand(1).item() > p:
        return X

    
    
    X = X.clone()
    batch_count, time_duration, feature_size = X.shape
    device = X.device

    #\no time steps=nothing to mask
    if time_duration == 0:
        return X

    # for each trial in batch... 
    for trial in range(batch_count):
        #choose random window width between 1 and min(max_width, time_duration)
        #never mask a window longer than the trial itself
        width = torch.randint(
            low=1,
            high=min(max_width, time_duration) + 1,
            size=(1,),
            device=device,
        ).item()

        #get valid starting index such that [start_index, start_index + width)
        #lies inside time axis [0, time_duration)
        if time_duration - width <= 0:
            #degenerate case: if width = time_duration,start at 0
            start_index = 0
        else:
            #sample start time from valid ranges
            start_index = torch.randint(
                low=0,
                high=time_duration - width + 1,
                size=(1,),
                device=device,
            ).item()

        end_index = start_index + width

        #zero out time window for all features
        #to simulate a short period of bad dtaa
        X[trial, start_index:end_index, :] = 0.0

    #time-masked batch
    return X


def time_feature_masking(X, max_time_width, electrode_frac=0.25, p=0.2):
    """
    Joint masking in time and features chooses a time window and a random subset
    of electrodes, and zeros out both TX and SBP channels for those electrodes
    only during that window. This models transient, localized corruption.
    """

    #with prob (1 - p) return X with no mods
    if torch.rand(1).item() > p:
        return X

    X = X.clone()
    batch_count, time_duration, feature_size = X.shape
    device = X.device

    #number of electrodes (TX and SBP pairs)
    num_electrodes = 128
    if time_duration == 0:
        return X

    num_mask_elec = max(1, int(num_electrodes * electrode_frac))

    #go through each trial in batch
    for trial in range(batch_count):
        #random time-window width [1, min(max_time_width, time_duration)]
        width = torch.randint(
            low=1,
            high=min(max_time_width, time_duration) + 1,
            size=(1,),
            device=device,
        ).item()

        #starting time index is s.t [start_index, start_index + width)
        #within [0, time_duration)
        if time_duration - width <= 0:
            start_index = 0
        else:
            start_index = torch.randint(
                low=0,
                high=time_duration - width + 1,
                size=(1,),
                device=device,
            ).item()

        end_index = start_index + width

        #sample subset of electrodes to mask
        #dont take same electrode >1time.
        elec_idx = torch.randperm(num_electrodes, device=device)[:num_mask_elec]

        
        #zero out TX
        X[trial, start_index:end_index, elec_idx] = 0.0
        # zero out SBP
        X[trial, start_index:end_index, elec_idx + num_electrodes] = 0.0

    return X

def time_shift(X, max_shift=1, p=0.1):
    """
    Randomly shift the entire time series forward or backward by a few bins, making
    the decoder bette with temporal misalignment between neural signals and targets.
    """

    if torch.rand(1).item() > p or max_shift <= 0:
        return X

    batch_count, time_duration, feature_size = X.shape
    device = X.device

    X_original = X.clone()
    X_shifted = torch.zeros_like(X_original)

    #for each trail in a batch
    for trial in range(batch_count):
        #pick randint shift in interva -max_shift, max_shift
        shift = torch.randint(
            low=-max_shift,
            high=max_shift + 1,
            size=(1,),
            device=device,
        ).item()

        if shift == 0:
            #no shift so do nothing
            X_shifted[trial] = X_original[trial]
        elif shift > 0:
            #positive shift to futrue
            X_shifted[trial, shift:, :] = X_original[trial, :-shift, :]
            X_shifted[trial, :shift, :] = 0.0
        else:
            #negative shift  into past
            s = -shift
            X_shifted[trial, :time_duration - s, :] = X_original[trial, s:, :]
            X_shifted[trial, time_duration - s:, :] = 0.0

    return X_shifted