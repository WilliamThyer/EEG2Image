import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.io as sio
from pathlib import Path

class Experiment:
    def __init__(self, data_dir, info_from_file = True, test = False):

        self.data_dir = Path(data_dir)

        self.xdata_files = list(self.data_dir.glob('*xdata*.mat'))
        self.ydata_files = list(self.data_dir.glob('*ydata*.mat'))
        if test:
            self.xdata_files = self.xdata_files[0:2]
            self.ydata_files = self.ydata_files[0:2]
        self.nsub = len(self.xdata_files)

        self.info_files = None

        if info_from_file:
            self.info = self.load_info(0)
            self.info.pop('unique_id')
            self.t = self.info['times']
            self.chan_x = self.info['chan_x']
            self.chan_y = self.info['chan_y']
            self.chan_labels = self.info['chan_labels']
            
    def load_eeg(self,isub):
        subj_mat = sio.loadmat(self.xdata_files[isub],variable_names=['xdata'])
        xdata = np.moveaxis(subj_mat['xdata'],[0,1,2],[1,2,0])

        subj_mat = sio.loadmat(self.ydata_files[isub],variable_names=['ydata'])
        ydata = np.squeeze(subj_mat['ydata'])

        return xdata, ydata
    
    def load_info(self, isub, variable_names = ['unique_id','chan_labels','chan_x','chan_y','chan_z','sampling_rate','times']):
        """ 
        loads info file that contains data about EEG file and subject
        """
        if not self.info_files:
            self.info_files = list(self.data_dir.glob('*info*.mat'))
        info_file = sio.loadmat(self.info_files[isub],variable_names=variable_names)
        info = {k: np.squeeze(info_file[k]) for k in variable_names}
        
        return info

    def moving_average(self, x, w):
        xnew = np.convolve(x, np.ones(w), 'same') / w
        return xnew[0:-4:5] ## very much hardcoded
    
    def prep_eeg(self, xdata):
        
        #chan regions
        upper = self.chan_y>0
        lower = self.chan_y<0
        right = self.chan_x>0
        left = self.chan_x<0
        region_idx = [upper&right,upper&left,lower&right,lower&left]

        # preallocating
        xregion = np.ones((xdata.shape[0],len(region_idx),xdata.shape[2]))
        newlen = self.moving_average(xdata[0,0,:],5).shape[0]
        xdata_new = np.ones((xdata.shape[0],len(region_idx),newlen))

        for ir,r in enumerate(region_idx):
            xregion[:,ir,:] = np.mean(xdata[:,r,:],1)
            for it in range(xdata_new.shape[0]):
                xdata_new[it,ir,:] = self.moving_average(xregion[it,ir,:],5)



class gaf:
    def init(self):
        pass

    def _create_time_serie(self, size, time):
        """Generate a time serie of length size and dynamic with respect to time."""
        # Generating time-series
        support = np.arange(0, size)
        serie = np.sin(support + float(time))
        return serie

    def tabulate(self, x, y, f):
        """Return a table of f(x, y). Useful for the Gram-like operations."""
        return np.vectorize(f)(*np.meshgrid(x, y, sparse=True))

    def cos_sum(self, a, b):
        """To work with tabulate."""
        return(math.cos(a+b))

    def transform(self, serie):
        """Compute the Gramian Angular Field of an image"""
        # Min-Max scaling
        min_ = np.amin(serie)
        max_ = np.amax(serie)
        scaled_serie = (2*serie - max_ - min_)/(max_ - min_)

        # Floating point inaccuracy!
        scaled_serie = np.where(scaled_serie >= 1., 1., scaled_serie)
        scaled_serie = np.where(scaled_serie <= -1., -1., scaled_serie)

        # Polar encoding
        phi = np.arccos(scaled_serie)

        # GAF Computation (every term of the matrix)
        gaf = self.tabulate(phi, phi, self.cos_sum)

        return gaf, scaled_serie

    def uni2gaf(self, serie=None,cmap=None):
        
        if serie is None:
            print("No time serie provided, using generated data!")
            serie = self._create_time_serie(45, 0)

        #transform and plot serie
        gaf, _ = self.transform(serie)
        if cmap is None:
            plt.matshow(gaf)
        else:
            plt.matshow(gaf,cmap=cmap)
        plt.axis('off')

    