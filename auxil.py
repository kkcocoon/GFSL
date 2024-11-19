import os
import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from operator import truediv
import random
from torch.utils.data.dataset import Dataset
import torch
import torch.nn.functional as functional

from sklearn.cluster import AgglomerativeClustering
import h5py

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# In[load hyperspectral dataset and create cubes according to patch size]:
def loadData(args):

   # directly load temp data from file if exists
   temp_data_path = os.path.join(os.getcwd(),'temp_data')
   temp_data_name = str(args.dataset) + str(args.components) + '_p'+str(args.patchsize) + '.h5'
   temp_data_name = os.path.join(temp_data_path,temp_data_name)

   if not os.path.isdir(temp_data_path):
      os.mkdir(temp_data_path)

   if os.path.exists(temp_data_name):
      print('Load the existing temp file:  '+temp_data_name)
      pf = h5py.File(temp_data_name, 'r')
      cubes = np.array(pf['cubes'])
      labels = np.array(pf['labels'])
      idx = np.array(pf['idx'])
      idy = np.array(pf['idy'])
      gt = np.array(pf['gt'])
      pf.close()    
      return cubes,labels,idx,idy,gt 
   
   # if temp data file not exists:
   print('Resetting the temp file:  '+temp_data_name)
   
   data_path = args.data_path
   dataset = args.dataset
   components = args.components
   if data_path==None:
      data_path = os.path.join(os.getcwd(),'data')
        
   if dataset == 'IP':
      data = sio.loadmat(os.path.join(data_path, 'indian_pines_corrected.mat'))['indian_pines_corrected']
      gt = sio.loadmat(os.path.join(data_path, 'indian_pines_gt.mat'))['indian_pines_gt']
   elif dataset == 'IP_Gabor':
      data = sio.loadmat(os.path.join(data_path, 'Indian16_gabor320.mat'))['data']
      gt = sio.loadmat(os.path.join(data_path, 'Indian_pines_gt.mat'))['indian_pines_gt']
   elif dataset == 'IP_EMAP':
      data = sio.loadmat(os.path.join(data_path, 'Indian_pines_corrected_EMAP.mat'))['indian_pines_corrected']
      gt = sio.loadmat(os.path.join(data_path, 'Indian_pines_gt.mat'))['indian_pines_gt']
   elif dataset == 'IP_EMAP85':
      data = sio.loadmat(os.path.join(data_path, 'Indian_pines_corrected_EMAP85.mat'))['indian_pines_corrected']
      gt = sio.loadmat(os.path.join(data_path, 'Indian_pines_gt.mat'))['indian_pines_gt']
   elif dataset == 'IP9_EMAP':
      data = sio.loadmat(os.path.join(data_path, 'indian_pines_corrected_EMAP.mat'))['indian_pines_corrected']
      gt = sio.loadmat(os.path.join(data_path, 'indian9_gt.mat'))['gt']
   elif dataset == 'IP9':
      data = sio.loadmat(os.path.join(data_path, 'indian_pines_corrected.mat'))['indian_pines_corrected']
      gt = sio.loadmat(os.path.join(data_path, 'indian9_gt.mat'))['gt']
   elif dataset == 'SV':
      data = sio.loadmat(os.path.join(data_path, 'salinas_corrected.mat'))['salinas_corrected']
      gt = sio.loadmat(os.path.join(data_path, 'salinas_gt.mat'))['salinas_gt']
   elif dataset == 'SV_EMAP':
      data = sio.loadmat(os.path.join(data_path, 'salinas_corrected_EMAP.mat'))['salinas_corrected']
      gt = sio.loadmat(os.path.join(data_path, 'salinas_gt.mat'))['salinas_gt']
   elif dataset == 'PU':
      data = sio.loadmat(os.path.join(data_path, 'paviaU.mat'))['paviaU']
      gt = sio.loadmat(os.path.join(data_path, 'paviaU_gt.mat'))['paviaU_gt']
   elif dataset == 'PU_EMAP':
      data = sio.loadmat(os.path.join(data_path, 'paviaU_EMAP.mat'))['paviaU']
      gt = sio.loadmat(os.path.join(data_path, 'paviaU_gt.mat'))['paviaU_gt']
   elif dataset == 'PC_EMAP':
      data = sio.loadmat(os.path.join(data_path, 'pavia_EMAP.mat'))['pavia']
      gt = sio.loadmat(os.path.join(data_path, 'pavia_gt.mat'))['pavia_gt']

   elif dataset == 'Botswana_EMAP':
      data = sio.loadmat(os.path.join(data_path,  'Botswana_EMAP.mat'))['Botswana']
      gt = sio.loadmat(os.path.join(data_path,  'Botswana_gt.mat'))['Botswana_gt']
        
   elif dataset == 'IP_100':
      data = sio.loadmat(os.path.join(data_path, 'IP_100.mat'))['indian_pines_corrected']
      gt = sio.loadmat(os.path.join(data_path, 'indian_pines_gt.mat'))['indian_pines_gt']

   elif dataset == 'SA_100':
      data = sio.loadmat(os.path.join(data_path, 'Salinas_100.mat'))['salinas_corrected']
      gt = sio.loadmat(os.path.join(data_path, 'Salinas_gt.mat'))['salinas_gt']
   elif dataset == 'PU_100':
      data = sio.loadmat(os.path.join(data_path, 'Paviau_100.mat'))['paviaU']
      gt = sio.loadmat(os.path.join(data_path, 'paviaU_gt.mat'))['paviaU_gt']        
   elif dataset == 'Chikusei_100':
      data = sio.loadmat(os.path.join(data_path, 'Chikusei_100.mat'))['chikusei']
      gt = sio.loadmat(os.path.join(data_path, 'Chikusei_gt.mat'))['chikusei_gt']
        
   elif dataset == 'IPS':
      data = sio.loadmat(os.path.join(data_path, 'data_UDA/IPS.mat'))['data']
      gt = sio.loadmat(os.path.join(data_path, 'data_UDA/IPS.mat'))['gt']
   elif dataset == 'IPT':
      data = sio.loadmat(os.path.join(data_path, 'data_UDA/IPT.mat'))['data']
      gt = sio.loadmat(os.path.join(data_path, 'data_UDA/IPT.mat'))['gt']
      
   elif dataset == 'PU7':
      data = sio.loadmat(os.path.join(data_path, 'data_UDA/paviaU.mat'))['paviaU']
      gt = sio.loadmat(os.path.join(data_path, 'data_UDA/paviaU_gt_7.mat'))['paviaU_gt_7']
   elif dataset == 'PC7':
      data = sio.loadmat(os.path.join(data_path, 'data_UDA/Pavia.mat'))['pavia']
      gt = sio.loadmat(os.path.join(data_path, 'data_UDA/pavia_gt_7.mat'))['pavia_gt_7']

   elif dataset == 'Houston13_7':
      data = sio.loadmat(os.path.join(data_path, 'data_UDA/Houston13.mat'))['ori_data']
      gt = sio.loadmat(os.path.join(data_path, 'data_UDA/Houston13_7gt.mat'))['map']
   elif dataset == 'Houston18_7':
      data = sio.loadmat(os.path.join(data_path, 'data_UDA/Houston18.mat'))['ori_data']
      gt = sio.loadmat(os.path.join(data_path, 'data_UDA/Houston18_7gt.mat'))['map']


   elif dataset == 'PU7_EMAP51': # UDA low accuracy for EMAP feature 
      data = sio.loadmat(os.path.join(data_path, 'data_UDA/paviaU_EMAP51.mat'))['paviaU']
      gt = sio.loadmat(os.path.join(data_path, 'data_UDA/paviaU_gt_7.mat'))['paviaU_gt_7']
   elif dataset == 'PC7_EMAP51':
      data = sio.loadmat(os.path.join(data_path, 'data_UDA/Pavia_EMAP51.mat'))['pavia']
      gt = sio.loadmat(os.path.join(data_path, 'data_UDA/pavia_gt_7.mat'))['pavia_gt_7']     
      
   elif dataset == 'PU7_EMAP170':
      data = sio.loadmat(os.path.join(data_path, 'data_UDA/paviaU_EMAP170.mat'))['paviaU']
      gt = sio.loadmat(os.path.join(data_path, 'data_UDA/paviaU_gt_7.mat'))['paviaU_gt_7']
   elif dataset == 'PC7_EMAP170':
      data = sio.loadmat(os.path.join(data_path, 'data_UDA/Pavia_EMAP170.mat'))['pavia']
      gt = sio.loadmat(os.path.join(data_path, 'data_UDA/pavia_gt_7.mat'))['pavia_gt_7']  
      
   elif dataset == 'KSC':
      data = sio.loadmat(os.path.join(data_path, 'KSC.mat'))['KSC']
      gt = sio.loadmat(os.path.join(data_path, 'KSC_gt.mat'))['KSC_gt']
   elif dataset == 'KSC_EMAP':
      data = sio.loadmat(os.path.join(data_path, 'KSC_EMAP.mat'))['KSC']
      gt = sio.loadmat(os.path.join(data_path, 'KSC_gt.mat'))['KSC_gt']
   elif dataset == 'Chikusei14_EMAP':
      data = sio.loadmat(os.path.join(data_path, 'Chikusei_EMAP.mat'))['chikusei']
      gt = sio.loadmat(os.path.join(data_path, 'Chikusei14_gt.mat'))['chikusei_gt']
   elif dataset.find('Chikusei')>-1:
      data = sio.loadmat(os.path.join(data_path, dataset+'.mat'))['chikusei']
      gt = sio.loadmat(os.path.join(data_path, 'Chikusei_gt.mat'))['chikusei_gt']
   elif dataset == 'HoustonU_EMAP':
      data = sio.loadmat(os.path.join(data_path,'HoustonU_EMAP.mat'))['houstonU']
      gt = sio.loadmat(os.path.join(data_path,'HoustonU_gt.mat'))['houstonU_gt']
   elif dataset.find('Houston')>-1:
      data = sio.loadmat(os.path.join(data_path,dataset+'.mat'))['Houston']
      gt = sio.loadmat(os.path.join(data_path,'Houston_gt.mat'))['Houston_gt']
      
   elif dataset == 'WHU_Hi_HanChuan_EMAP':
      data = sio.loadmat(os.path.join(data_path, 'WHU_Hi_HanChuan_EMAP.mat'))['WHU_Hi_HanChuan']
      gt = sio.loadmat(os.path.join(data_path, 'WHU_Hi_HanChuan_gt.mat'))['WHU_Hi_HanChuan_gt']
   elif dataset == 'WHU_Hi_HanChuan_EMAP85':
      data = sio.loadmat(os.path.join(data_path, 'WHU_Hi_HanChuan_EMAP85.mat'))['WHU_Hi_HanChuan']
      gt = sio.loadmat(os.path.join(data_path, 'WHU_Hi_HanChuan_gt.mat'))['WHU_Hi_HanChuan_gt']
   elif dataset == 'WHU_Hi_HanChuan_EMAP_Train25':
      data = sio.loadmat(os.path.join(data_path, 'WHU_Hi_HanChuan_EMAP.mat'))['WHU_Hi_HanChuan']
      gt = sio.loadmat(os.path.join(data_path, 'WHU-Hi-HanChuan-Training-test/Train25.mat'))['Train25']
   elif dataset == 'WHU_Hi_HanChuan_EMAP_Test25':
      data = sio.loadmat(os.path.join(data_path, 'WHU_Hi_HanChuan_EMAP.mat'))['WHU_Hi_HanChuan']
      gt = sio.loadmat(os.path.join(data_path, 'WHU-Hi-HanChuan-Training-test/Test25.mat'))['Test25']
   elif dataset == 'WHU_Hi_LongKou_EMAP':
      data = sio.loadmat(os.path.join(data_path, 'WHU_Hi_LongKou_EMAP.mat'))['WHU_Hi_LongKou']
      gt = sio.loadmat(os.path.join(data_path, 'WHU_Hi_LongKou_gt.mat'))['WHU_Hi_LongKou_gt']
   elif dataset == 'WHU_Hi_HongHu_EMAP':
      data = sio.loadmat(os.path.join(data_path, 'WHU_Hi_HongHu_EMAP.mat'))['WHU_Hi_HongHu']
      gt = sio.loadmat(os.path.join(data_path, 'WHU_Hi_HongHu_gt.mat'))['WHU_Hi_HongHu_gt']

   elif dataset == 'WHU_Hi_LongKou_EMAP85':
      data = sio.loadmat(os.path.join(data_path, 'WHU_Hi_LongKou_EMAP85.mat'))['WHU_Hi_LongKou']
      gt = sio.loadmat(os.path.join(data_path, 'WHU_Hi_LongKou_gt.mat'))['WHU_Hi_LongKou_gt']
   elif dataset == 'WHU_Hi_HongHu_EMAP85':
      data = sio.loadmat(os.path.join(data_path, 'WHU_Hi_HongHu_EMAP85.mat'))['WHU_Hi_HongHu']
      gt = sio.loadmat(os.path.join(data_path, 'WHU_Hi_HongHu_gt.mat'))['WHU_Hi_HongHu_gt']
      
   else:
      print("NO DATASET")
      exit()
      
   shapeor = data.shape
      
   data = data.reshape(-1, data.shape[-1])
   if components != None:
      data = PCA(n_components=components).fit_transform(data)
      shapeor = np.array(shapeor)
      shapeor[-1] = components
   #data = MinMaxScaler().fit_transform(data)  
   data = StandardScaler().fit_transform(data)  # X = (X-X_mean)/X_std
   data = data.reshape(shapeor)
   gt = gt.astype(np.uint16)
   
   #return data, labels, num_class

   cubes,labels,idx,idy = createImageCubes(data, gt, patchsize=args.patchsize)
   
   # Save data to temp file for next fast loading
   pf = h5py.File(temp_data_name, 'w')  
   pf['cubes'] = cubes
   pf['labels'] = labels
   pf['idx'] = idx
   pf['idy'] = idy
   pf['gt'] = gt
   pf.close()
   
   return cubes,labels,idx,idy,gt


# In[Generate data loader to get training batch]:

def load_hyper(args):
   
   if isinstance(args.dataset,list):
      datasets = args.dataset
      i=0
      for datai in datasets:
         args.dataset = datai
         if i==0:
            cubes,labels,idx,idy,gt = loadData(args)            
         else:
            mlab = max(labels)+1
            tcubes,tlabels,tidx,tidy,gt = loadData(args)
            tlabels = tlabels + mlab
            
            cubes = np.concatenate((cubes,tcubes))                 
            labels = np.concatenate((labels,tlabels))
            idx = np.concatenate((idx,tidx))
            idy = np.concatenate((idy,tidy))
         i=i+1
      args.dataset = datasets
   else:
      cubes,labels,idx,idy,gt = loadData(args)
      
   bands = cubes.shape[-1]; numberofclass = len(np.unique(labels))
   shape = cubes.shape[0:2];
   if args.tr_percent < 1: # split by percent
      x_train, x_test, y_train, y_test, idx_te, idy_te = split_data(cubes, labels,args.tr_percent, idx=idx,idy=idy,rand_state=args.rand_state)
   else: # split by samples per class
      x_train, x_test, y_train, y_test, idx_te, idy_te = split_data_fix(cubes, labels, args.tr_percent, idx=idx,idy=idy,rand_state=args.rand_state)
   if args.use_val: 
      if args.val_percent<1: # noted: x_test0 and x_test
         x_val, x_test0, y_val, y_test0, _, _ = split_data(x_test, y_test, args.val_percent, idx=idx,idy=idy,rand_state=args.rand_state)
      else:
         x_val, x_test0, y_val, y_test0, _, _ = split_data_fix(x_test, y_test, args.val_percent, idx=idx,idy=idy,rand_state=args.rand_state)
   del cubes
   train_hyper = HyperData((np.transpose(x_train, (0, 3, 1, 2)).astype("float32"),y_train))
   
   test_hyper = None
   if args.use_test: test_hyper  = HyperData((np.transpose(x_test, (0, 3, 1, 2)).astype("float32"),y_test))
   
   val_hyper = None
   if args.use_val: val_hyper = HyperData((np.transpose(x_val, (0, 3, 1, 2)).astype("float32"),y_val))
   
   #kwargs = {'num_workers': 1, 'pin_memory': True}
   kwargs = {'num_workers': 0, 'pin_memory': True}
   
   if args.tr_bsize>len(x_train):
      args.tr_bsize=len(x_train)
   train_loader = None
   if len(x_train)>0:
      #train_loader = torch.utils.data.DataLoader(train_hyper, batch_size=args.tr_bsize, shuffle=args.shuffle_train, drop_last=True,**kwargs)
      train_loader = torch.utils.data.DataLoader(train_hyper, batch_size=args.tr_bsize, shuffle=args.shuffle_train, drop_last=True,**kwargs)
   
   if args.use_test: 
      test_loader  = torch.utils.data.DataLoader(test_hyper, batch_size=args.te_bsize, shuffle=False, **kwargs)
   else:
      test_loader = None
      
   if args.use_val:
      val_loader  = torch.utils.data.DataLoader(val_hyper, batch_size=args.te_bsize, shuffle=False, **kwargs)
   else:
      val_loader  = None
   return train_loader, test_loader, val_loader, numberofclass, bands, shape, idx_te, idy_te, gt


class HyperData(Dataset):
    def __init__(self, dataset):
        self.data = dataset[0].astype(np.float32)
        self.labels = []
        for n in dataset[1]: self.labels += [int(n)]

    def __getitem__(self, index):
        img = torch.from_numpy(np.asarray(self.data[index,:,:,:]))
        label = self.labels[index]
        return img, label

    def __len__(self):
        return len(self.labels)

    def __labels__(self):
        return self.labels
         
# In[create cubes]:

def createImageCubes(data, gt, patchsize=11, removeZeroLabels=True, half=True, ridx=True, step=1):
   if removeZeroLabels:
      (tidx,tidy) = np.where(gt>0)  # 0 means background
   else:
      (tidx,tidy) = np.where(gt>-1)
   
   # if there are too much samples, take part of samples for experiment
   if len(tidx)>50000 and half: 
      np.random.seed(123);
      rind = np.random.permutation(len(tidx))
      rind = rind[:len(rind)//4]
      tidx = tidx[rind]
      tidy = tidy[rind]
      
   margin = int((patchsize - 1) / 2)
   dsize = len(np.arange(0,patchsize,step)) # downsample by step
   X_pad = np.lib.pad(data,((margin,margin),(margin,margin),(0,0)),'symmetric') # 对称边界延拓，而不是补0
   patchesData = np.zeros((len(tidx), dsize, dsize, data.shape[2]))
   patchesLabels = np.zeros((len(tidx),))
   tidx = tidx + margin
   tidy = tidy + margin
   pi=0
   for r,c in zip(tidx,tidy):
      patch = X_pad[r-margin:r+margin+1:step, c-margin:c+margin+1:step]   
      patchesData[pi, :, :, :] = patch
      patchesLabels[pi] = gt[r-margin, c-margin]
      pi=pi+1
   
   patchesLabels = patchesLabels.astype(np.int32)-1  # without background, labels start from 0
    
#   if removeZeroLabels:
#      rind = np.random.permutation(len(patchesLabels))
#      patchesData = patchesData[rind]
#      patchesLabels = patchesLabels[rind].astype("int")
#      tidx = tidx[rind]
#      tidy = tidy[rind]
   
   idx = tidx - margin
   idy = tidy - margin
   
   if ridx:
      return patchesData, patchesLabels,idx,idy
   else:
      return patchesData, patchesLabels

# In[randomly select data]:
   
def random_unison(a,b, rstate=None):
   assert len(a) == len(b)
   p = np.random.RandomState(seed=rstate).permutation(len(a))
   return a[p], b[p]


def split_data_fix(cubes, labels, n_samples, idx=None,idy=None,rand_state=None):
   train_set_size = [n_samples] * len(np.unique(labels))
   return split_data(cubes, labels, 0, train_set_size, idx,idy,rand_state)

def split_data(cubes, labels, percent, train_set_size=None, idx=None,idy=None,rand_state=None):
   
   cubes_number = np.unique(labels, return_counts=1)[1]

   if train_set_size is None or len(train_set_size)!=len(cubes_number):
      train_set_size = [int(np.ceil(a*percent)) for a in cubes_number]

   # Prealloc memory, faster
   tr_size = int(sum(train_set_size)) # 0
   te_size = int(sum(cubes_number)) # 0
   
   sizetr = np.array([tr_size]+list(cubes.shape)[1:])
   sizete = np.array([te_size]+list(cubes.shape)[1:])
   train_x = np.empty((sizetr)); train_y = np.empty((tr_size),dtype=np.int16); test_x = np.empty((sizete)); test_y = np.empty((te_size),dtype=np.int16)
   idx_te = np.empty((sizete[0]))
   idy_te = np.empty((sizete[0]))

   tr_count = 0
   tt_count = 0
   for cl in np.unique(labels):
      
      bind = np.where(labels==cl)[0]
      cubes_cl = cubes[bind]
      labels_cl = labels[bind]
      
      # If there are not enough samples for class cl, without using the setting 'n_samples'.
      tlen = len(bind)
      if tlen<train_set_size[cl]:
         #train_set_size[cl] = int(0.75*tlen) # take 75% for training
         train_set_size[cl] = tlen           # take all for training
         
      pind = np.random.RandomState(seed=rand_state).permutation(tlen)
      trind = pind[0:train_set_size[cl]] # random select samples
      ttind = pind[train_set_size[cl]:]
 
      train_x[tr_count:tr_count+len(trind)] = cubes_cl[trind]
      train_y[tr_count:tr_count+len(trind)] = labels_cl[trind]
      tr_count = tr_count+len(trind)
      test_x[tt_count:tt_count+len(ttind)] = cubes_cl[ttind]
      test_y[tt_count:tt_count+len(ttind)] = labels_cl[ttind]
      idx_te[tt_count:tt_count+len(ttind)] = idx[bind[ttind]]
      idy_te[tt_count:tt_count+len(ttind)] = idy[bind[ttind]]

      tt_count = tt_count+len(ttind)
      # print(bind[pind[0:5]])
      #print(trind)
   
   train_x = train_x[0:tr_count]
   train_y = train_y[0:tr_count]
   test_x = test_x[0:tt_count]
   test_y = test_y[0:tt_count]
   idx_te = idx_te[0:tt_count]
   idy_te = idy_te[0:tt_count]

   # train_x, train_y = random_unison(train_x, train_y, rstate=rand_state)
   return train_x, test_x, train_y, test_y, idx_te, idy_te

        


# In[Evalate result]:
      
def predict(testloader, model):
   model.eval()
   preds = []
   labels = []
   feas=[]
   for batch_idx, (test_x, test_y) in enumerate(testloader):
      test_x = test_x.to(device)
      fea, pred = model(test_x)
      [preds.append(a) for a in pred.data.cpu().numpy()] 
      [labels.append(a) for a in test_y] 
      [feas.append(a.detach().cpu().numpy()) for a in fea] 
   feas = np.array(feas)
   preds_mx = np.array(preds)
   preds = np.argmax(preds_mx, axis=1)
   return preds, preds_mx, np.array(labels),feas

def predict2(testloader, model, classifier1, classifier2):
   model.eval()
   preds = []
   labels = []
   feas=[]
   for batch_idx, (test_x, test_y) in enumerate(testloader):
      test_x = test_x.to(device)
      fea = model(test_x)
      pred = classifier1(fea) + classifier2(fea)
      [preds.append(a) for a in pred.data.cpu().numpy()] 
      [labels.append(a) for a in test_y] 
      [feas.append(a.detach().cpu().numpy()) for a in fea] 
   feas = np.array(feas)
   preds_mx = np.array(preds)
   preds = np.argmax(preds_mx, axis=1)
   return preds, preds_mx, np.array(labels),feas


def accuracy(output, target, topk=(1,)):
   """Computes the precision@k for the specified values of k"""
   maxk = max(topk)
   batch_size = target.size(0)

   _, pred = output.topk(maxk, 1, True, True)
   pred = pred.t()
   correct = pred.eq(target.view(1, -1).expand_as(pred))

   res = []
   for k in topk:
      correct_k = correct[:k].view(-1).float().sum(0)
      res.append(correct_k.mul_(100.0 / batch_size))
   return res

def get_accuracy(logits, targets):
    _, predictions = torch.max(logits, dim=-1)
    return torch.mean(predictions.eq(targets).float())

def AA_andEachClassAccuracy(confusion_matrix):
   #counter = confusion_matrix.shape[0]
   list_diag = np.diag(confusion_matrix)
   list_raw_sum = np.sum(confusion_matrix, axis=1)
   each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
   average_acc = np.mean(each_acc)
   return each_acc, average_acc


def reports(y_pred, y_test):
   classification = classification_report(y_test, y_pred)
   oa = accuracy_score(y_test, y_pred)
   confusion = confusion_matrix(y_test, y_pred)
   each_acc, aa = AA_andEachClassAccuracy(confusion)
   kappa = cohen_kappa_score(y_test, y_pred)

   return classification, confusion, list(np.round(np.array([oa, aa, kappa] + list(each_acc)) * 100, 2))


def cdd(output_t1,output_t2):   
   output_t1 = functional.softmax(output_t1,dim=1) 
   output_t2 = functional.softmax(output_t2,dim=1) 
 
   mul = output_t1.transpose(0, 1).mm(output_t2)
   cdd_loss = torch.sum(mul) - torch.trace(mul)
   return cdd_loss
 
# In[t-SNE visualize]:
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

# We import seaborn to make nice plots.
import seaborn as sns
# sns.set_style('darkgrid')
# sns.set_palette('muted')
# sns.set_context("notebook", font_scale=1.5,
#                 rc={"lines.linewidth": 2.5})

def plot_TSNE(X,y,cc=None, picname=None):
#   tsne_obj = TSNE(random_state=2020).fit(X)
#   digits_proj = tsne_obj.transform(X)
   digits_proj = TSNE(random_state=2020).fit_transform(X)
   f, ax, sc, txts = scatter(digits_proj, y)
   plt.scatter
   if cc:
      xx = digits_proj[-cc:,0]
      yy = digits_proj[-cc:,1]
      ax.scatter(xx,yy,c='r',marker='o')
      
      for i in range(cc):
         ax.text(xx[i], yy[i], str(i+1), fontsize=20, c='r')
   plt.show()

   if picname:
      plt.savefig(picname, dpi=120)
   

def scatter(x, y):
    NumClass = max(y)+1    
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", NumClass))

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
                    c=palette[y.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    for i in range(NumClass):
        # Position of each label.
        xtext, ytext = np.median(x[y == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i+1), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts


def same_seeds(seed):
   torch.manual_seed(seed)
   if torch.cuda.is_available():
      torch.cuda.manual_seed(seed)
      torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
   np.random.seed(seed)  # Numpy module.
   random.seed(seed)  # Python random module.
   torch.backends.cudnn.benchmark = False
   torch.backends.cudnn.deterministic = True

#from sklearn.datasets import load_digits
#digits = load_digits()
## We first reorder the data points according to the handwritten numbers.
#X = np.vstack([digits.data[digits.target==i]
#               for i in range(10)])
#y = np.hstack([digits.target[digits.target==i]
#               for i in range(10)])
#plot_TSNE(X,y)   