import torch
import torch.nn as nn
import torch.nn.functional as F
# from efficientnet_pytorch import EfficientNet
import math
import timm
from einops import rearrange
from  swin_transformer_modify import SwinTransformer
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        # padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        # self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class CSAB(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CSAB, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.channel_attention(x)
        out = out * self.spatial_attention(out)
        return out






class HyperStructure2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HyperStructure2, self).__init__()
        self.hyper_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, stride=1, padding=0),
           
            nn.GELU(),
            nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(in_channels // 2, out_channels, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        return self.hyper_block(x)

    
class FeatureNet(nn.Module):
    def __init__(self):
        super(FeatureNet, self).__init__()
    
      
        pretrained_cfg_overlay = {'file': r"/home/user/Vit_Add_Eff/pytorch_model_Effv2m21k.bin"}
        self.RGBNet = timm.create_model('tf_efficientnetv2_m.in21k', pretrained_cfg_overlay=pretrained_cfg_overlay,
                                        features_only=True, pretrained=True)


        model = SwinTransformer(img_size=384,patch_size=4, window_size=12, embed_dim=48, depths=(2, 2, 6, 2), num_heads=[3,6,12,24],num_classes=1)

      
        
        self.swmodel=model
        
         
       
        self.conv1=nn.Conv2d(in_channels=48, out_channels=48,kernel_size=3, stride=1,padding=1)
        self.conv2=nn.Conv2d(in_channels=80, out_channels=96,kernel_size=3, stride=1,padding=1)
        self.conv3=nn.Conv2d(in_channels=176, out_channels=192,kernel_size=3, stride=1,padding=1)
        self.conv4=nn.Conv2d(in_channels=512, out_channels=384,kernel_size=3, stride=1,padding=1)
        
        
        self.gap1 = nn.AdaptiveAvgPool2d((1, 1))
   
        self.CSAB1 = CSAB(384)
        self.CSAB2 = CSAB(384)
        self.CSAB3 = CSAB(384)
        self.CSAB4 = CSAB(384)
        self.CSABsvit = CSAB(384)
        self.CSABCNN = CSAB(512)
     

        self.norm_layer=nn.LayerNorm(512)
       
        self.hyper1_1_rgb = HyperStructure2(48, 80) 
        self.hyper1_2_rgb = HyperStructure2(80, 176)  
        self.hyper1_3_rgb = HyperStructure2(176, 512) 

        self.hyper2_1_rgb = HyperStructure2(80, 176)  
        self.hyper2_2_rgb = HyperStructure2(176, 512)  

        self.hyper3_1_rgb = HyperStructure2(176, 512) 
        
        
    
        
    def forward(self, input1):
        input_rgb = input1.view(-1, input1.size(-3), input1.size(-2), input1.size(-1))
       
        endpoints_rgb = self.RGBNet(input_rgb)
      
      
        a0_rgb = endpoints_rgb[0]
        a1_rgb = endpoints_rgb[1]  
        a2_rgb = endpoints_rgb[2]  
        a3_rgb = endpoints_rgb[3]  
        a4_rgb = endpoints_rgb[4]  
     
        
     
        rgb_hyper1 = self.hyper1_1_rgb(a1_rgb)  
        
       
        rgb_hyper1 = self.hyper1_2_rgb(rgb_hyper1 + a2_rgb)  
        rgb_hyper2 = self.hyper2_1_rgb(a2_rgb) 
        
       
        rgb_hyper1 = self.hyper1_3_rgb(rgb_hyper1 + a3_rgb)  
        rgb_hyper2 = self.hyper2_2_rgb(rgb_hyper2 + a3_rgb) 
        rgb_hyper3 = self.hyper3_1_rgb(a3_rgb) 
        
        F_CNN = rgb_hyper1 + rgb_hyper2 + rgb_hyper3 + a4_rgb  
       
     
      
       
        a1_rgb=self.conv1(a1_rgb)  
        a2_rgb=self.conv2(a2_rgb) 
        a3_rgb=self.conv3(a3_rgb)  
        a4_rgb=self.conv4(a4_rgb)  
        
       

       
        f1 = rearrange(a1_rgb, 'b c h w -> b (h w) c')
        f2 = rearrange(a2_rgb, 'b c h w -> b (h w) c')
        f3 = rearrange(a3_rgb, 'b c h w -> b (h w) c')
        
     
        
        F_CNN = rearrange(F_CNN, 'b c h w -> b (h w) c')
        F_CNN=self.norm_layer(F_CNN)
        
        f1,f2,f3=self.swmodel(input_rgb,f1,f2,f3)
       
   
        f1 = rearrange(f1, 'b (h w) c -> b c h w', h=12, w=12)  
        f2 = rearrange(f2, 'b (h w) c -> b c h w', h=12, w=12) 
        f3 = rearrange(f3, 'b (h w) c -> b c h w', h=12, w=12)  
        
        F_CNN = rearrange(F_CNN, 'b (h w) c -> b c h w', h=12, w=12) 
       
      
        
        f1=self.CSAB1(f1)
        f2=self.CSAB2(f2)
        f3=self.CSAB3(f3)
       
        F_CNN = self.CSABCNN(F_CNN)
        
        F_combined = torch.cat((f1,f2,f3,F_CNN), 1)
        
        F_combined = self.gap1(F_combined) 
        F_combined = F_combined.view(F_combined.size(0), -1)
      
        return F_combined


class FCNet(nn.Module):
    def __init__(self):
        super(FCNet, self).__init__()
        self.fc1 = nn.Linear(1664, 512)  
        self.fc_end=nn.Linear(512,1)
    

    def forward(self, x1):   
        out=  self.fc1(x1)
        out=self.fc_end(out)
        return out



class Net(nn.Module):
    def __init__(self, headnet, net):
        super(Net, self).__init__()
        self.headnet = headnet
        self.net = net
    def forward(self, x1):
        f1 = self.headnet(x1)
        output = self.net(f1)
        return output
#if __name__ == '__main__':
    #net1 = FeatureNet()
    #net2 = FCNet()
    #model = Net(headnet=net1, net=net2)

    #input = torch.randn((1, 3, 384, 384))

    #f1 = model(input)
    #print(f1.shape)


