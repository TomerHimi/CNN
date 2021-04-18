#Machine Learning Assignment 5
#DEVELOPED BY Tomer Himi & CUCUMBER AN OrSN COMPANY.
#UNAUTHORIZED COPY OF THIS WORK IS STRICTLY PROHIBITED.
#DEVELOPED FOR EDUCATIONAL PURPOSES, FOR THE COURSE MACHINE LEARNING 89511.
#BAR ILAN UNIVERSITY, JANUARY, 2021.
#ALL RIGHTS RESERVED.

import os
import os.path
import soundfile as sf
import librosa
import torch.utils.data as data
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_ as xavier

#Gcommand_dataset.py code
AUDIO_EXTENSIONS = ['.wav', '.WAV',]

def is_audio_file(filename):
    return any(filename.endswith(extension) for extension in AUDIO_EXTENSIONS)

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def make_dataset(dir, class_to_idx):
    spects = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_audio_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    spects.append(item)
    return spects

def spect_loader(path, window_size, window_stride, window, normalize, max_len = 101):
    y, sr = sf.read(path)
    #n_fft = 4096
    n_fft = int(sr * window_size)
    win_length = n_fft
    hop_length = int(sr * window_stride)

    #STFT
    D = librosa.stft(y, n_fft = n_fft, hop_length = hop_length,
                     win_length = win_length, window = window)
    spect, phase = librosa.magphase(D)

    #S = log(S+1)
    spect = np.log1p(spect)

    #make all spects with the same dims
    if spect.shape[1] < max_len:
        pad = np.zeros((spect.shape[0], max_len - spect.shape[1]))
        spect = np.hstack((spect, pad))
    elif spect.shape[1] > max_len:
        spect = spect[:, :max_len]
    spect = np.resize(spect, (1, spect.shape[0], spect.shape[1]))
    spect = torch.FloatTensor(spect)

    #z-score normalization
    if normalize:
        mean = spect.mean()
        std = spect.std()
        if std != 0:
            spect.add_(-mean)
            spect.div_(std)
    return spect

class GCommandLoader(data.Dataset):
    """A google command data set loader where the wavs are arranged in this way: ::
        root/one/xxx.wav
        root/one/xxy.wav
        root/one/xxz.wav
        root/head/123.wav
        root/head/nsdf3.wav
        root/head/asd932_.wav
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        window_size: window size for the stft, default value is .02
        window_stride: window stride for the stft, default value is .01
        window_type: typye of window to extract the stft, default value is 'hamming'
        normalize: boolean, whether or not to normalize the spect to have zero mean and one std
        max_len: the maximum length of frames to use
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        spects (list): List of (spects path, class_index) tuples
        STFT parameter: window_size, window_stride, window_type, normalize
    """

    def __init__(self, root, transform = None, target_transform = None, window_size = .02, #Constructor of class GCommand
                 window_stride = .01, window_type = 'hamming', normalize = True, max_len = 101):
        classes, class_to_idx = find_classes(root)
        spects = make_dataset(root, class_to_idx)

        if len(spects) == 0:
            raise (RuntimeError("Found 0 sound files in subfolders of: " + root + "Supported audio file extensions are: " + ",".join(AUDIO_EXTENSIONS)))

    #Fields
        self.root = root
        self.spects = spects
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = spect_loader
        self.window_size = window_size
        self.window_stride = window_stride
        self.window_type = window_type
        self.normalize = normalize
        self.max_len = max_len
        self.len = len(self.spects)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (spect, target) where target is class_index of the target class.
        """
        path, target = self.spects[index]
        spect = self.loader(path, self.window_size, self.window_stride, self.window_type, self.normalize, self.max_len)
        if self.transform is not None:
            spect = self.transform(spect)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return spect, target

    def __len__(self):
        return self.len
    
class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace = True),
            nn.Conv2d(2, 4, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Conv2d(4, 8, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace = True),
            nn.Conv2d(8, 16, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2), 
            nn.Conv2d(16, 32, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2),)

        self.linear_layers = nn.Sequential(
            nn.Linear(15360, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(inplace=True),
            nn.Linear(50, 30),)
          
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return F.log_softmax(x)        

def train(model, train_loader):
    """Train function, trains our model using the ordinary drill of Forward, Backwards etc.
    After each epoch of training, validation step has activated"""
    model.train()

    for batch_idx, (data_, labels) in enumerate(train_loader):  #train step
        optimizer.zero_grad()
        output = model(data_)  # performs Forward apparently
        loss = F.nll_loss(output, labels)
        loss.backward()
        optimizer.step()

def validation(model, validation_loader, epoch):
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data_, labels in validation_loader:  #val step
            output = model(data_)
            val_loss += F.nll_loss(output, labels, reduce = True).item() #sum up batch loss
            pred = output.max(1, keepdim = True)[1]  #get the index of the max log-probability
            correct += pred.eq(labels.view_as(pred)).cpu().sum()

    val_loss /= len(validation_loader.dataset)
    val_accuracy = 100. * correct / len(validation_loader.dataset)
    print("\tValidation set: Epoch: {}, Accuracy: {:.2f}%".format(epoch, val_accuracy))

def test(model, test_loader):
    """Test function, helps us predict the label of each example of fashion MNIST (test from PyTorch)"""
    files_names = []
    dirs = test_loader.dataset.root
    for target in sorted(os.listdir(dirs)):
        d = os.path.join(dirs, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for k, fname in enumerate(sorted(fnames)):
                string = fname
                files_names.append(string)
    files_names.remove('.DS_Store')
    
    index = 0
    correct = 0  
    final_pred = {}
    label_y = []
    total = []
    
    model.eval()
    with torch.no_grad():
        for data_, target in test_loader:  #test step
            output = model(data_)
            pred = output.max(1, keepdim = True)[1]
            correct += pred.eq(target.view_as(pred)).cpu().sum()
            for label in pred:    
                label_y.append(label.item())
            final_pred[files_names[index]] = newDictionary.get(label_y[index])
            index += 1
    
    sorted_files = sorted(final_pred, key = lambda x: int(x.split('.')[0]))
    for i in sorted_files:
        total.append(i + ',' + final_pred[i])
    np.savetxt("test_y", total, fmt = "%s", delimiter = "")          
    test_accuracy = correct / len(test_loader.dataset)
  
def weight_init(m):
   if isinstance(m, nn.Conv2d):
      xavier(m.weight.data)

if __name__ == "__main__":

    datasetOfTrain = GCommandLoader(r'gcommands/train')  #Converting audio files to images.
    train_loader = torch.utils.data.DataLoader(datasetOfTrain, batch_size = 10, shuffle = True, pin_memory = True)
    
    datasetOfValid = GCommandLoader(r'gcommands/valid')
    validation_loader = torch.utils.data.DataLoader(datasetOfValid, batch_size = 10, shuffle = True, pin_memory = True)

    datasetOfTest = GCommandLoader(r'gcommands/test')
    test_loader = torch.utils.data.DataLoader(datasetOfTest, shuffle = False, pin_memory = True)

    newDictionary = dict([(value, key) for key, value in datasetOfTrain.class_to_idx.items()])
    model = Network()
    model.apply(weight_init)  #apply is a saved term in Python which refer to another function that is passed in the argument.
    #In other words, there ISN'T a function named "apply", rather a function named weight_init.
    #https://python-reference.readthedocs.io/en/latest/docs/functions/apply.html
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)  # lr stands for Learning Rate. torch.optim injects an OPTIMIZER.
    epoch = 10

    #train and validation steps
    for num_epoch in range(epoch):
        train(model, train_loader)  
        validation(model, validation_loader, num_epoch) 
    test(model, test_loader)  #Outside of FOR LOOP deliberately. 