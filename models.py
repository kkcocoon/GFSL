import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# In[proposed loss]:

# Kernel similarity. the bigger, the more similar.
def kernel_metric(a,b,use_kernel=True,kernel_num=5,kernel_mul=2.0):
    n = a.shape[0]
    m = b.shape[0]
    #m = len(b)
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = ((a - b) ** 2).sum(dim=2)

    if not use_kernel:
       return -logits  # Euclidean similarity. Note that there is a minus sign
    
    n_samples = int(n)+int(m)
    bandwidth = torch.sum(logits.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [ torch.exp(-logits / bandwidth_temp) for bandwidth_temp in bandwidth_list]

    return sum(kernel_val)
 
# Samples should be arranged according to categories     
class KernelFewShotLoss(nn.Module):
    def __init__(self, class_num, use_kernel=True, kernel_num=1):
        super(KernelFewShotLoss, self).__init__()
        self.class_num = class_num
        self.use_kernel = use_kernel
        self.kernel_num = kernel_num
        self.criterion_CE_pair = torch.nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        """
        Args:
           inputs: feature matrix with shape (batch_size, feat_dim)
           targets: ground truth labels with shape (batch_size,),  not one-hot martix
        Note:
           Targets should be arranged according to classes!!!
           Each class should contain the same number of samples!!!
        """
        num = inputs.size(0)
        num_per_class = int(num/self.class_num)

        class_mean = inputs.reshape(self.class_num, num_per_class, -1).mean(dim=1)
        metric = kernel_metric(inputs, class_mean, self.use_kernel, self.kernel_num)
        loss = self.criterion_CE_pair(metric, targets)
      
        return loss
     
# Another KernelFewShot implement: Samples do NOT need to be arranged according to categories      
class KernelFewShotLoss1(nn.Module):
    def __init__(self, class_num, use_kernel=True, kernel_num=1):
        super(KernelFewShotLoss1, self).__init__()
        self.class_num = class_num
        self.use_kernel = use_kernel
        self.kernel_num = kernel_num
        self.criterion_CE_pair = torch.nn.CrossEntropyLoss()

    def forward(self, inputs, targets):

        # If the elements in the list contain tensors, 
        # it is impossible to directly convert them into a single tensor using torch.tensor(). 
        # Instead, you should first convert all the elements of the list into NumPy arrays. 
        # Or you can use torch.cat to concatenate each tensor, 
        # but be careful to expand the dimension by using unsqueeze(0) first.
        for c in range(self.class_num):
            tm = inputs[targets==c,:].mean(dim=0)
            tm = tm.unsqueeze(0)   # cast torch.Size([256]) to torch.Size([1, 256])
            if c==0:
               class_mean = tm
            else:
               class_mean = torch.cat((class_mean,tm),0)
            
        metric = kernel_metric(inputs, class_mean, self.use_kernel, self.kernel_num)
        loss = self.criterion_CE_pair(metric, targets)
      
        return loss    
     
# In[self-designed feature encoder network architecture]:

class MyNetwork(nn.Module):
    def __init__(self, input_channel, class_num, feature_dim=256):
        super(MyNetwork, self).__init__()

        self.conv11 = nn.Conv2d(input_channel, 128, 1, padding=0)
        self.bn11 = nn.BatchNorm2d(128, affine=True)
        
        self.conv12 = nn.Conv2d(128, 128, 1, padding=0)
        self.bn12 = nn.BatchNorm2d(128, affine=True)

        self.conv1 = nn.Conv2d(128, 128, 3, padding=0)
        self.bn1 = nn.BatchNorm2d(128, affine=True)

        self.conv2 = nn.Conv2d(128, 128, 3, padding=0)
        self.bn2 = nn.BatchNorm2d(128, affine=True)

        self.conv3 = nn.Conv2d(128, 64, 1, padding=0)
        self.bn3 = nn.BatchNorm2d(64, affine=True)

        self.fc1 = nn.Linear(1600, feature_dim)  # according to xfc.size()

        self.classfifer = nn.Linear(feature_dim, class_num)

    def forward(self, x, domain='target'):

        # 9x9 x 128
        x = self.conv11(x)
        x = self.bn11(x)
        x = F.relu(x)

        # 9x9 x 128
        x = self.conv12(x)
        x = self.bn12(x)
        x = F.relu(x)
      
        # 9x9 x 128
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        # 7x7 x 128
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        # 5x5 x 128
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        # 5x5 x 64
        xfc = x.view(x.size(0), -1)
        # print(xfc.size())  # to set the first dimension of self.fc1

        features = self.fc1(xfc)   # (batch_size, feature_dim)
        # print(features.size())

        output = self.classfifer(features) #(batch_size, target_class)

        return features, output
     
 