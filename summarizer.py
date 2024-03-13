import librosa
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
# The Audio object allows us to play back sound
# directly in the web browser
from IPython.display import Audio
from matplotlib import patches



def moving_average(x, w):
    """Computes moving average of window size w on x."""
    return np.convolve(x, np.ones(w), 'valid') / w



class SongSummary():
    
    def __init__(self,path, sr = 22050):
        if type(path)=='str':
            self.raw_song, self.sr = librosa.load(path)  
        elif type(path) == np.ndarray:
            self.raw_song = path
            self.sr = sr
        else: 
            raise('Path is either string to file audio or np.ndarray.')
        self.song, self.trim_idx  = librosa.effects.trim(self.raw_song)
        self.sim_matrix = None
        
    def compute_similarity(self, method, similarity, is_similarity=True, win_size=2048, window='hamming', summary_time=10):
        self.summary_time = summary_time # in seconds
        self.summary = {}
        self.win_size = win_size # in data-points
        self.win_seconds = self.win_size / self.sr # window size in seconds
        self.effective_win_size = int(self.win_size - self.win_size/8) # Hardcoded because of overlap hyperparam
        self.win_step_seconds = self.effective_win_size / self.sr # size of one step, consider overlap
        self.summary_size= summary_time * self.sr // self.effective_win_size#self.win_size # in windows

        # Define which side of the argsort will be selected as a (dis)similar
        if is_similarity:
            self.similar = -1
            self.disimilar = 0
        else:
            self.similar = 0
            self.disimilar = -1
        
        match method:
            case 'mfcc':
                melspec = librosa.feature.melspectrogram(y=self.song,hop_length=win_size, window='hamming',win_length=win_size)
                mfcc_features = librosa.feature.mfcc(y=self.song, sr=22050, S=melspec,n_mfcc=45, n_fft = win_size, window="hamming", win_length=win_size, hop_length=win_size)
                mfcc_var = np.std(mfcc_features,axis=1)**2
                args_mfcc_var = np.argsort(mfcc_var)
                mfcc_selected = mfcc_features[args_mfcc_var[-15::1],:].T
                self.features = mfcc_selected / np.std(mfcc_selected,axis=0)
            case 'spectrogram':
                freqs, times, sxx = signal.spectrogram(x=self.song, fs=22050, window='hamming', nperseg=win_size)
                self.features = sxx.T
            case 'norm-spectrogram':
                freqs, times, sxx = signal.spectrogram(x=self.song, fs=22050, window='hamming', nperseg=win_size)
                prefeatures = sxx.T
                self.features =  prefeatures / np.std(prefeatures,axis=0)
            case _:
                raise 'method must be one of [mfcc, spectrogram]'
        self.sim_matrix = similarity(self.features)


    
    def _get_ticks_and_labels(self, extend=False):
        
        if not extend:
            duration = len(self.sim_matrix)+1
        else:
            duration = int(self.point_to_second(len(self.song))/self.win_step_seconds)+1
            #step = 10
        step = int(10/self.win_step_seconds)
        
        ticks = [_+1 for _ in range(0,duration,step)] 
        
        labels = [0]
        labels.extend(list(int(self.window_to_second(_)) for _ in ticks[1:]))
        return ticks, labels
    
    def plot_matrix(self, highlight=False, triple=False):
        f, ax = plt.subplots()
        img = ax.imshow(self.sim_matrix)
        ticks, labels = self._get_ticks_and_labels()
        
        ax.set_yticks(ticks,labels)
        ax.set_xticks(ticks,labels)
        
        plt.colorbar(img)
        if highlight:
            if not triple:
                ax = plt.gca()
                # Create a Rectangle patch
                h = self.sim_matrix.shape[0] * self.summary_size/(len(self.song)/self.win_size)
                rect = patches.Rectangle((0, self.best-2), self.sim_matrix.shape[0],h+2, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
        plt.show()
        
    def plot_song(self, highlight=False, triple=False):
        f, ax = plt.subplots()
        songplot = ax.plot(self.song)
        if self.sim_matrix:
            ticks, labels = self._get_ticks_and_labels()
            ax.set_xticks(ticks,labels)
        
        if highlight:
            ax = plt.gca()
            if not triple:
                # Create a Rectangle patch
                ax.axvspan(self.start,self.end , alpha=0.25, color='green', )
            else:
                ax.axvspan(self.start_similar,self.start_similar+self.song_step , alpha=0.25, color='green', )
                ax.axvspan(self.start_disimilar,self.start_disimilar+self.song_step, alpha=0.25, color='green', )
                ax.axvspan(self.second_similar,self.second_similar+self.song_step, alpha=0.25, color='green', )

        plt.show()

    def plot_scores(self, extend=False, highlight=False):
        
        f, ax = plt.subplots()
        
        # Dealing with ticks and labels
        ticks, labels = self._get_ticks_and_labels(extend)
        if not extend:
            scores_to_plot = self.sliding_score
        else:            
            scores_to_plot = np.pad(self.sliding_score,(0,int(self.second_to_window(self.point_to_second(len(self.song)))) - len(self.sliding_score)), 'constant', constant_values=(0, 0))

        ax.set_xticks(ticks,labels)
        ax.plot(scores_to_plot)
        
        
        # Highlight summary section
        if highlight:
            raise('NotImplementedError')
               
        plt.show()

    def compute_scores(self, sliding_size, similarities):
        sim_matrix_j = np.mean(similarities,axis=1)
        self.sliding_score = moving_average(sim_matrix_j, sliding_size)
        self._m = np.mean(self.sliding_score)
        self._s = np.std(self.sliding_score)
        self.sorted = np.argsort(self.sliding_score)
        
    def summarize_cooperfoote(self): 
        self.compute_scores(self.summary_size, self.sim_matrix)
        self.best = self.sorted[self.similar]
        self.start = self.best*int(self.win_size-self.win_size/8) # accounts for overlapping
        self.end = self.start+self.summary_size*self.win_size
        self.summary.update({'cooper-foote':self.song[self.start:self.end]})

    def summarize_triple(self):
        self.sliding_size = int(self.summary_size/3)
        self.compute_scores(self.sliding_size, self.sim_matrix)
        self.song_step = self.sliding_size*self.win_size
        
        dis_upper_lim = int(len(self.sim_matrix) * 0.80) #- 10 * self.sliding_size
        sim_upper_lim = len(self.sim_matrix) - self.sliding_size

        # Compute the indexes
        self.dissimilar_index = self.sorted[np.where(self.sorted < dis_upper_lim)][self.disimilar]
        similar_begin = self.sorted[np.where(self.sorted < self.dissimilar_index)][self.similar]
        similar_end = self.sorted[np.where((self.sorted > self.dissimilar_index + self.sliding_size) & (self.sorted < sim_upper_lim) & (self.sliding_score > self._m - 1.5 * self._s))][self.similar]

        # Compute transform to points in the sample
        self.start_similar = similar_begin * (self.win_size-256)
        self.start_disimilar =  self.dissimilar_index  * (self.win_size-256)
        self.second_similar = similar_end * (self.win_size-256)
        self.summary.update({'triple':np.concatenate(
            (self.song[self.start_similar:self.start_similar+self.song_step],
            self.song[self.start_disimilar:self.start_disimilar+self.song_step],
            self.song[self.second_similar:self.second_similar+self.song_step]))
                            })
    def get_best(self):
        return self.best
    
    def play_song(self):
        return Audio(self.song, rate=self.sr)
    
    def play_summary(self, type):
        '''Type one of <cooper-foote> or <triple>.'''
        return Audio(self.summary[type],rate=self.sr)

    def point_to_second(self, point):
        return point / self.sr 
        
    def second_to_point(self, time):
        return time * self.sr

    def second_to_window(self, time):
        return time * self.sr / self.effective_win_size 
        
    def window_to_second(self, window):
        return window * self.effective_win_size / self.sr 