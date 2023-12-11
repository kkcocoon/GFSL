import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import argparse
import datetime

from models import MyNetwork
from models import KernelFewShotLoss

import numpy as np
import scipy.io as sio
import auxil

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# In[parser]:

parser = argparse.ArgumentParser(description='main')
args_main = parser.parse_args()    
  
# ========================================================================  
def main(args_main):
   
   parser = argparse.ArgumentParser(description='PyTorch Training')
   parser.add_argument('--episodes', default=1000, type=int, help='(2000) Number of total episodes to run')
   parser.add_argument("-f","--feature_dim",type = int, default = 256)
   parser.add_argument('--lr', '--learning_rate', default=0.001, type=float, help='(0.001) Important! Initial learning rate')
   
   #parser.add_argument('--data_path', default=r'E:\6.PythonPro\10.Group Experiment\datasets-EMAPorg', type=str, help='data path')
   # parser.add_argument('--data_path', default=r'/opt/data/private/data_HSI', type=str, help='data path')
   parser.add_argument('--data_path', default=r'G:\data_HSI', type=str, help='data path')
   
   parser.add_argument('--components', default=None, type=int, help='dimensionality reduction')
   parser.add_argument('--dataset', default='IP_EMAP', type=str, help='dataset (options: IP_EMAP, SV_EMAP, PU_EMAP, IP, PU, SV)')
   parser.add_argument('--tr_percent', default=5, type=float, help='(5 or 0.05) Samples of train set')
   parser.add_argument('--tr_bsize', default=1000, type=int, help='few shot: large enough such that a batch contains all training samples . ')
   parser.add_argument('--te_bsize', default=128, type=int, help='(128) Mini-batch test size')
   parser.add_argument('--use_test',default=True, type=bool, help='Use test set')
   parser.add_argument('--use_val',default=True, type=bool, help='Use validation set')
   parser.add_argument('--val_percent', default=0.1, type=float, help='(0.05) samples of val set')
   parser.add_argument('--shuffle_train', default=False, type=int, help='')
   
   parser.add_argument('--use_kernel', default=True, type=bool, help='')
   parser.add_argument('--kernel_num', default=5, type=int, help='')   
   
   parser.add_argument('--loss_hard_margin', default=2.0, type=int, help='')   
   
   
   parser.add_argument('--patchsize',  default=9, type=int, help='spatial patch size')
   parser.add_argument('--optimizer',  default='SGD', type=str, help='optimizer')
   
   parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
   parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, help='(1e-4) Weight decay ')
   
   parser.add_argument('--loss_metric_weight', default=5, type=float, help='Important! Weight for triplet loss')
   
   parser.add_argument('--net', default='MyNetwork', type=str, help='(pRestNet, MyNetwork) Net model')
   parser.add_argument('--rand_state', default=1331, type=int, help='(None,123) Random seed')
   
   # Ensure the results for different runnings are the same for the same random seed.
   auxil.same_seeds(0)
   
   args = parser.parse_args()
   
   args_dict = vars(args)
   args_main_dict = vars(args_main)
   for key in args_main_dict.keys():
      args_dict[key] = args_main_dict[key]
   args = argparse.Namespace(**args_dict)
   
   #for k in args_main_dict.keys():
   #   print(k,':', args_dict[k])
   
   #for key in args_dict.keys():
   #   print(key,':', args_dict[key])
   
   
   # In[load data and create network]:
   
   print("\n******Load data******")
   #args.dataset='IP_EMAP'
   #args.tr_percent = 5
   #args.val_percent = 0.1 
   #args.use_val = True
   #args.use_test = True
   args.shuffle_train = False # important! proposed loss requires samples to be arranged according to categories
   train_loader, test_loader, val_loader, class_num, band_num, target_shape, idx_te, idy_te, gt = auxil.load_hyper(args)
   
   num_train=0; num_test=0; num_val=0;
   if train_loader is not None: num_train = train_loader.dataset.__len__()
   if test_loader is not None: num_test = test_loader.dataset.__len__()
   if val_loader is not None: num_val = val_loader.dataset.__len__()
   print('Target: {}, Class:{}, Train:{}, Val:{}, Test:{}\n'.format(args.dataset, class_num, num_train,num_val,num_test))
   
   print("Training episodes = {}\n".format(args.episodes))
   
   feature_encoder = MyNetwork(band_num,class_num, args.feature_dim)
   feature_encoder = feature_encoder.to(device)
   
   # In[train]:
   
   # nn.CrossEntropyLoss() = nn.logSoftmax() + nn.NLLLoss()
   # loss_CE = -(target_one_hot * torch.log_softmax(logits, dim=-1)).sum(dim=-1)
   criterion_CE = torch.nn.CrossEntropyLoss().to(device)
   
   criterion_KFSL = KernelFewShotLoss(class_num,args.use_kernel,args.kernel_num).to(device)
   
   feature_encoder_optim = torch.optim.SGD(feature_encoder.parameters(), lr=args.lr, momentum=args.momentum,
                     weight_decay=args.weight_decay, nesterov=True)
   
   #   # load previous model to continue training
   #   if os.path.exists("last_model.pth"): 
   #      checkpoint = torch.load("last_model.pth")
   #      feature_encoder.load_state_dict(checkpoint['state_dict'])
   
   # few labeled samples for training
   train_iter = iter(train_loader)     
   
   start_time=datetime.datetime.now()
   
   test_acc = -1
   best_acc = -1
   best_episode = args.episodes
   best_results = [0,0,0]
   #best_acc_class = []
   last_results = [0,0,0]
   last_acc_class = []
   
   for episode in range(args.episodes):
   
      # ----------------------------------------------------------------------------
      '''get a batch data'''
         
      # Because batch size is larger than the total number of training samples,
      # the batch samples for different iters are the same, i.e. all the training samples
      try:
         train, train_label = train_iter.next()
      except Exception as err:
         train_iter = iter(train_loader) 
         train, train_label = train_iter.next()
   
      train = train.to(device)
      train_label = train_label.to(device) # should be arranged according to classes!!!
      
      # ----------------------------------------------------------------------------
      '''compute the loss and train the network'''
      feature_encoder.train()
      train_feature, train_output = feature_encoder(train)
      loss_metric = criterion_KFSL(train_feature,train_label)
      loss_ce = criterion_CE(train_output, train_label)
      loss = args.loss_metric_weight * loss_metric + loss_ce
   
      feature_encoder.zero_grad()
      loss.backward()
      feature_encoder_optim.step()
   
      # ----------------------------------------------------------------------------
      '''display results at some episodes'''
      if (episode + 1) % 100 == 0:
         feature_encoder.eval()
   
         train_feature, train_output = feature_encoder(train)
         tr_acc = auxil.accuracy(train_output, train_label)[0].item()
   
         print('e:{}, metric: {:.4f}, CE: {:.4f}, total: {:.2f},  tr_acc:{:.2f}'.format(
                 episode + 1,
                 loss_metric.item(),
                 loss_ce.item(),
                 loss.item(),
                 tr_acc))
   
      if (episode + 1) % 200 == 0 or episode == 0:
         if not args.use_val:
            continue
         
         preds,preds_mx,labels,feas = auxil.predict(val_loader, feature_encoder)
         classification, confusion, results = auxil.reports(preds,labels)
         test_acc = results[0]
   
         if test_acc >= best_acc:
            state = {
                  'args_dict': args_dict,
                  'best_acc': test_acc,
                  'state_dict': feature_encoder.state_dict(),
                  'optimizer' : feature_encoder_optim.state_dict(),
            }
            torch.save(state, "best_model.pth")
            best_episode = episode
            best_acc = test_acc
            best_results = results
            #best_acc_class = np.diag(confusion) / np.sum(confusion, 1, dtype=np.float)
   
         print('episode:{}, acc = {:.2f};  best_episode:{}, best_acc = {:.2f}'.format(episode + 1, test_acc, best_episode + 1, best_acc))
   
   # In[record]:   
   state = {
      'args_dict': args_dict,
      'best_acc': test_acc,
      'state_dict': feature_encoder.state_dict(),
      'optimizer': feature_encoder_optim.state_dict(),
   }
   torch.save(state, "last_model.pth")
   
   duration_tr = datetime.datetime.now()-start_time
   
   if args.use_val and args.use_test:
      checkpoint = torch.load("best_model.pth")
      feature_encoder.load_state_dict(checkpoint['state_dict'])
      preds,preds_mx,labels,feas = auxil.predict(test_loader, feature_encoder)
      classification, confusion, best_results = auxil.reports(preds,labels)
      #best_acc_class = np.diag(confusion) / np.sum(confusion, 1, dtype=np.float)
   if args.use_test:
      checkpoint = torch.load("last_model.pth")      
      feature_encoder.load_state_dict(checkpoint['state_dict'])
      preds,preds_mx,labels,feas = auxil.predict(test_loader, feature_encoder)
      classification, confusion, last_results = auxil.reports(preds,labels)
      last_acc_class = np.diag(confusion) / np.sum(confusion, 1, dtype=np.float)
   
   fname = 'train/{}_{}_{}'.format(str(args.dataset),args.tr_percent,args.rand_state)
   fname = fname + '_metric_weight_{}_episode_{}'.format(args.loss_metric_weight,best_episode+1)
   
   #gt = gt+1
   #results_test = {'pred': preds, 'test_label': labels, 'idx': idx_te, 'idy': idy_te, 'gt':gt}
   #fname = fname + '_{:.2f}'.format(last_results[0])
   #np.savez(fname, results_test)
   #   
   #for ij in np.arange(0, len(preds)):
   #   x = int(idx_te[ij])
   #   y = int(idy_te[ij])
   #   gt[x, y] = preds[ij]+1
   #data = {
   #   'data': gt
   #}
   #sio.savemat(fname+'.mat', data)
   #plt.figure(dpi=200)
   #plt.imshow(gt)
   
   #sresult = '{:.2f}, {:.2f} # {}-trtime_{}'.format(last_results[0], best_results[0], fname, duration_tr.seconds)
   sresult = '{:.2f}, {:.2f}, {:.2f} # {}-trtime_{}'.format(last_results[0], last_results[1], last_results[2], fname, duration_tr.seconds)
   file_handle = open('result_KFSL.txt','a')
   file_handle.write('\n')
   file_handle.writelines(sresult)
   file_handle.close()
   print(sresult)
   
   return last_results, last_acc_class, best_results 
   
if __name__ == '__main__':
   	main(args_main)