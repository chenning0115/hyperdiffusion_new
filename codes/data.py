import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import torch
import torch.nn as nn
import torch.optim as optim
from operator import truediv
import time

""" Training dataset"""

class TrainDS(torch.utils.data.Dataset):

    def __init__(self, Xtrain, ytrain):

        self.len = Xtrain.shape[0]
        self.x_data = torch.FloatTensor(Xtrain)
        self.y_data = torch.LongTensor(ytrain)

    def __getitem__(self, index):

        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]
    def __len__(self):

        # 返回文件数据的数目
        return self.len

""" Testing dataset"""

class TestDS(torch.utils.data.Dataset):

    def __init__(self, Xtest, ytest):

        self.len = Xtest.shape[0]
        self.x_data = torch.FloatTensor(Xtest)
        self.y_data = torch.LongTensor(ytest)

    def __getitem__(self, index):

        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]

    def __len__(self):

        # 返回文件数据的数目
        return self.len



class HSIDataLoader(object):
    def __init__(self, param) -> None:
        self.data_param = param['data']
        self.data = None #原始读入X数据 shape=(h,w,c)
        self.labels = None #原始读入Y数据 shape=(h,w,1)
        
        self.X = None
        self.Y = None

        # 参数设置
        self.data_sign = self.data_param.get('data_sign', 'Indian')

    def load_data(self):
        data, labels = None, None
        if self.data_sign == "Indian":
            data = sio.loadmat('../data/Indian_pines_corrected.mat')['indian_pines_corrected']
            labels = sio.loadmat('../data/Indian_pines_gt.mat')['indian_pines_gt']
        elif self.data_sign == "piava":
            pass
        else:
            pass
        return data, labels

    def _padding(self, X, margin=2):
        # pading with zeros
        w,h,c = X.shape
        new_x, new_h, new_c = w+margin*2, h+margin*2, c
        returnX = np.zeros((new_x, new_h, new_c))
        start_x, start_y = margin, margin
        returnX[start_x:start_x+w, start_y:start_y+h,:] = X
        return returnX

    def padding_side_one(self, X):
        w,h,c = X.shape
        returnX = np.zeros((128,128,c))
        returnX[:, :,:] = X[:128,:128,:]
        return returnX
        
    def applyPCA(self, X, numComponents=30):
        newX = np.reshape(X, (-1, X.shape[2]))
        pca = PCA(n_components=numComponents, whiten=True)
        newX = pca.fit_transform(newX)
        newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
        return newX

    def generate_torch_dataset(self):
        #1. 根据data_sign load data
        self.data, self.labels = self.load_data()

        #1.1 norm化
        norm_data = np.zeros(self.data.shape)
        for i in range(self.data.shape[2]):
            input_max = np.max(self.data[:,:,i])
            input_min = np.min(self.data[:,:,i])
            norm_data[:,:,i] = (self.data[:,:,i]-input_min)/(input_max-input_min)
        
        if self.data_param.get('pca', 0) > 0:
            norm_data = self.applyPCA(norm_data, int(self.data_param['pca']))

        print('[data] load data shape data=%s, label=%s' % (str(norm_data.shape), str(self.labels.shape)))

        # norm_data = self.padding_side_one(norm_data)
        
        #4. 调整shape来满足torch使用
        X_all = norm_data.transpose((2, 0, 1))
        X_all = np.expand_dims(X_all, 0)
        Y_all = np.expand_dims(self.labels, 0)
        # Y_all = self.labels
        print('------[data] after transpose train, test------')
        print("Y_test shape : %s" % str(X_all.shape))

        X = TrainDS(X_all, Y_all)
        self.X = X_all
        self.Y = Y_all
        print('------final--------')
        print('X.shape=', X_all.shape)
        print('Y.shape=', Y_all.shape)
        all_data_loader = torch.utils.data.DataLoader(dataset=X,
                                                    batch_size=1,
                                                    shuffle=False,
                                                    num_workers=0,
                                                )
        return all_data_loader

    def get_X_Y(self):
        return self.X, self.Y



if __name__ == "__main__":
    dataloader = HSIDataLoader({'data':{}})
    all_dataloader = dataloader.generate_torch_dataset()