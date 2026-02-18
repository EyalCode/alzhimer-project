import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models
from torch.autograd import Variable
from pointnet2_utils import PointNetSetAbstraction, PointNetSetAbstractionMsg
from torchvision.models import resnet50, ResNet50_Weights,convnext_tiny, ConvNeXt_Tiny_Weights



class TNet(nn.Module):
    def __init__(self, k=64):
        super(TNet, self).__init__()

        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNet2D(nn.Module):
    def __init__(self, num_classes=10, dropout=0.33):
        super(PointNet2D, self).__init__()
        
        # Input transformation network
        self.input_transform = TNet(k=2)
        
        # First shared MLP
        self.conv1 = nn.Conv1d(2, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        
        # Feature transformation network
        self.feature_transform = TNet(k=64)
        
        # Second shared MLP
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)
        
        # Global feature MLP
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)
        self.bn6 = nn.BatchNorm1d(512)
        self.bn7 = nn.BatchNorm1d(256)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: batch_size x 2 x num_points
        batch_size = x.size(0)
        num_points = x.size(2)
        
        # Input transformation
        trans = self.input_transform(x)
        #x = x.transpose(2, 1)
        #x = torch.bmm(x, trans)
        #x = x.transpose(2, 1)

        # First shared MLP
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        # Feature transformation
        trans_feat = self.feature_transform(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans_feat)
        x = x.transpose(2, 1)
        
        # Second shared MLP
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        
        # Max pooling
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        
        # MLP for classification
        x = F.relu(self.bn6(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn7(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x
    

class PointNetPlusPlus(nn.Module):
    def __init__(self,num_class,normal_channel=False):
        super(PointNetPlusPlus, self).__init__()
        in_channel = 2 if normal_channel else 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(512, [0.2, 0.4, 0.8], [16, 32, 128], in_channel,[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 640, [256, 512, 1024], True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_class)

    def forward(self, xy):
        B, _, _ = xy.shape
        if self.normal_channel:
            norm = xy[:, 2:, :]
            xy = xy[:, :2, :]
        else:
            norm = None
        l1_xy, l1_points = self.sa1(xy, norm)
        l2_xy, l2_points = self.sa2(l1_xy, l1_points)
        l3_xy, l3_points = self.sa3(l2_xy, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)


        return x,l3_points
        

class PointNetResNetFusion(nn.Module):
    def __init__(self, num_class, normal_channel=False):
        super(PointNetResNetFusion, self).__init__()

        in_channel = 2 if normal_channel else 0
        self.normal_channel = normal_channel
        
        self.sa1 = PointNetSetAbstractionMsg(512, [0.2, 0.4, 0.8], [16, 32, 128], in_channel,[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 640, [256, 512, 1024], True)
        

        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)

        self.resnet.fc = nn.Identity()

        for param in self.resnet.parameters():
            param.requires_grad = True
        
        self.fc1 = nn.Linear(3072, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        
        self.fc3 = nn.Linear(256, num_class)

    def forward(self, xy, img):

        B, _, _ = xy.shape
        if self.normal_channel:
            norm = xy[:, 2:, :]
            xy = xy[:, :2, :]
        else:
            norm = None
            
        l1_xy, l1_points = self.sa1(xy, norm)
        l2_xy, l2_points = self.sa2(l1_xy, l1_points)
        l3_xy, l3_points = self.sa3(l2_xy, l2_points)

        point_feats = l3_points.view(B, 1024)

        img_feats = self.resnet(img)
        combined_feats = torch.cat((point_feats, img_feats), dim=1)

        x = self.drop1(F.relu(self.bn1(self.fc1(combined_feats))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)

        return x, combined_feats
        
        
        
class PointNetConvNextFusion(nn.Module):
    def __init__(self, num_class, normal_channel=False):
        super(PointNetConvNextFusion, self).__init__()

        in_channel = 2 if normal_channel else 0
        self.normal_channel = normal_channel
        
        self.sa1 = PointNetSetAbstractionMsg(512, [0.2, 0.4, 0.8], [16, 32, 128], in_channel,[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 640, [256, 512, 1024], True)

        self.cnn = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)

        self.cnn.classifier = nn.Identity()

        for param in self.cnn.parameters():
            param.requires_grad = True
            

        
        self.fc1 = nn.Linear(1792, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        
        self.fc3 = nn.Linear(256, num_class)

    def forward(self, xy, img):

        B, _, _ = xy.shape
        if self.normal_channel:
            norm = xy[:, 2:, :]
            xy = xy[:, :2, :]
        else:
            norm = None
            
        l1_xy, l1_points = self.sa1(xy, norm)
        l2_xy, l2_points = self.sa2(l1_xy, l1_points)
        l3_xy, l3_points = self.sa3(l2_xy, l2_points)
        
        # Flatten PointNet output: [Batch, 1024]
        point_feats = l3_points.view(B, 1024)

        img_feats = self.cnn(img).flatten(1) 
        combined_feats = torch.cat((point_feats, img_feats), dim=1)

        x = self.drop1(F.relu(self.bn1(self.fc1(combined_feats))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)

        return x, combined_feats
        
        
class PointNetConvNextFusionBase(nn.Module):

    def __init__(self, num_class=250, normal_channel=False): 
        super(PointNetConvNextFusionBase, self).__init__()
        
        in_channel = 2 if normal_channel else 0
        self.normal_channel = normal_channel

        self.sa1 = PointNetSetAbstractionMsg(512, [0.2, 0.4, 0.8], [16, 32, 128], in_channel,[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 640, [256, 512, 1024], True)
        
        self.pointnet_output_dim = 1024 

        print("Loading ConvNeXt Base weights...")
        self.cnn = models.convnext_base(weights='DEFAULT') 
        self.cnn.classifier = nn.Identity()
        self.cnn_output_dim = 1024 

        fusion_input_dim = self.pointnet_output_dim + self.cnn_output_dim
        
        self.fusion_fc = nn.Sequential(
            nn.Linear(fusion_input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(512, num_class)
        )

    def forward(self, point_input, img_input):

        xy = point_input
        B, _, _ = xy.shape
        
        if self.normal_channel:
            norm = xy[:, 2:, :]
            xy = xy[:, :2, :]
        else:
            norm = None
            
        l1_xy, l1_points = self.sa1(xy, norm)
        l2_xy, l2_points = self.sa2(l1_xy, l1_points)
        l3_xy, l3_points = self.sa3(l2_xy, l2_points)

        point_feats = l3_points.view(B, 1024)
        img_feats = self.cnn(img_input)
        img_feats = img_feats.view(img_feats.size(0), -1)

        combined = torch.cat((point_feats, img_feats), dim=1)
        output = self.fusion_fc(combined)
        
        return output