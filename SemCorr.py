import os
import copy
from parser import get_args

import cv2
import numpy as np
from makeup import Makeup
from PIL import Image

import torch
import random
import functools
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.models as models

class LeakyReLUConv2d(nn.Module):
    def __init__(self, inplanes, outplanes, kernel_size, stride, padding, norm='None', sn=False):
        super(LeakyReLUConv2d, self).__init__()
        model = []
        model += [nn.ReflectionPad2d(padding)]
        if sn:
            model += [nn.utils.spectral_norm(
                nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size, stride=stride, padding=0, bias=True))]
        else:
            model += [nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size, stride=stride, padding=0, bias=True)]
        if norm == 'Instance':
            model += [nn.InstanceNorm2d(outplanes, affine=False)]
        model += [nn.LeakyReLU(0.2, inplace=False)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
    
class Upsample(nn.Module):
    """upsample Block with conditional instance normalization."""

    def __init__(self, in_channels, out_channels, is_up=True):
        super(Upsample, self).__init__()
        self.is_up = is_up
        if self.is_up:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.actv = nn.LeakyReLU(0.2, inplace=False)

    def forward(self, x):
        if self.is_up:
            x = self.up(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.actv(x)
        return x


class ResBlock(nn.Module):
    """Residual Block with conditional instance normalization."""

    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm2d(channels)
        self.nn1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.InstanceNorm2d(channels)

    def forward(self, x):
        y = self.conv1(x)
        y = self.norm1(y)
        y = self.nn1(y)
        y = self.conv2(y)
        y = self.norm2(y)
        return x + y

class Encoder_Semantic(nn.Module):
    def __init__(self, parse_dim, ngf=32):
        super(Encoder_Semantic, self).__init__()
        self.parse_dim = parse_dim
        self.conv1 = LeakyReLUConv2d(parse_dim, ngf * 1, kernel_size=7, stride=1, padding=3, norm='instance')
        self.conv2 = LeakyReLUConv2d(ngf * 1, ngf * 2, kernel_size=3, stride=2, padding=1, norm='instance')
        self.conv3 = LeakyReLUConv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1, norm='instance')
        self.conv4 = LeakyReLUConv2d(ngf * 4, ngf * 8, kernel_size=3, stride=2, padding=1, norm='instance')
        self.conv5 = LeakyReLUConv2d(ngf * 8, ngf * 16, kernel_size=3, stride=2, padding=1, norm='instance')
        self.res1 = ResBlock(channels=ngf * 16)
        self.res2 = ResBlock(channels=ngf * 16)
        self.up1 = Upsample(in_channels=ngf * 16, out_channels=ngf * 8, is_up=True)
        self.up2 = Upsample(in_channels=ngf * 8, out_channels=ngf * 4, is_up=True)
        # self.up3 = Upsample(in_channels=ngf * 4, out_channels=ngf * 4, is_up=True)
        # self.up4 = Upsample(in_channels=ngf * 4, out_channels=ngf * 4, is_up=True)

    def forward(self, parse):
        ins_feat = parse 

        x_range = torch.linspace(-1, 1, ins_feat.shape[-1], device=ins_feat.device)
        y_range = torch.linspace(-1, 1, ins_feat.shape[-2], device=ins_feat.device)
        y, x = torch.meshgrid(y_range, x_range)  
        y = y.expand([ins_feat.shape[0], 1, -1, -1]) 
        x = x.expand([ins_feat.shape[0], 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)  
        input = torch.cat([ins_feat, coord_feat], 1) 
        # print(input.shape)
        output1 = self.conv1(input)
        output2 = self.conv2(output1)
        output3 = self.conv3(output2)
        output4 = self.conv4(output3)
        output5 = self.conv5(output4)
        output = self.res1(output5)
        output = self.res2(output)
        output = self.up1(output+output5)
        output = self.up2(output + output4)
        # output = self.up4(self.up3(output))
        return output
    
class Attention(nn.Module):
    def __init__(self, channels, norm='Instance', sn=False):
        super(Attention, self).__init__()
        in_dim = channels
        self.chanel_in = in_dim
        self.softmax_alpha = 100
        self.eps = 1e-5
        self.fa_conv = LeakyReLUConv2d(in_dim, in_dim // 4, kernel_size=1, stride=1, padding=0, norm=norm, sn=sn)
        self.fb_conv = LeakyReLUConv2d(in_dim, in_dim // 4, kernel_size=1, stride=1, padding=0, norm=norm, sn=sn)
        self.fc_conv1 = LeakyReLUConv2d(3, 3, kernel_size=3, stride=2, padding=1, norm=norm, sn=sn)
        self.fc_conv2 = LeakyReLUConv2d(3, 3, kernel_size=3, stride=2, padding=1, norm=norm, sn=sn)
        self.fc_up1 = Upsample(3, 3, is_up=True)
        self.fc_up2 = Upsample(3, 3, is_up=True)

    def cal_correlation(self, fa, fb, alpha):
        '''
            calculate correspondence matrix and warp the exemplar features
        '''
        assert fa.shape == fb.shape, \
            'Feature shape must match. Got %s in a and %s in b)' % (fa.shape, fb.shape)
        n, c, h, w = fa.shape
        # subtract mean
        fa = fa - torch.mean(fa, dim=(2, 3), keepdim=True)
        fb = fb - torch.mean(fb, dim=(2, 3), keepdim=True)

        # vectorize (merge dim H, W) and normalize channelwise vectors
        fa = fa.view(n, c, -1)
        fb = fb.view(n, c, -1)
        fa = fa / (torch.norm(fa, dim=1, keepdim=True) + self.eps)
        fb = fb / (torch.norm(fb, dim=1, keepdim=True) + self.eps)

        energy_ab_T = torch.bmm(fb.transpose(-2, -1), fa) * alpha
        corr_ab_T = F.softmax(energy_ab_T, dim=1)  # n*HW*C @ n*C*HW -> n*HW*HW
        return corr_ab_T

    def forward(self, fa_raw, fb_raw, fc_raw):
        fa = self.fa_conv(fa_raw)
        fb = self.fb_conv(fb_raw)
        # fc = F.interpolate(fc_raw,scale_factor=0.25,mode='bilinear')
        fc = self.fc_conv2(self.fc_conv1(fc_raw))
        corr_ab_T = self.cal_correlation(fa, fb, self.softmax_alpha)
        n, c, h, w = fc.shape
        fc_warp = torch.bmm(fc.view(n, c, h * w), corr_ab_T)  # n*HW*1
        fc_warp = fc_warp.view(n, c, h, w)
        fc_warp = self.fc_up2(self.fc_up1(fc_warp))
        return fc_warp, corr_ab_T
    
class Semantic_Correspodance_Module(nn.Module):
    def __init__(self, parse_dim, mx_channel):
        super(Semantic_Correspodance_Module, self).__init__()
        self.enc = Encoder_Semantic(parse_dim, mx_channel)
        self.attn = Attention(channels= mx_channel * 4)
        self.mx_channel = mx_channel
        self.parse_dim = parse_dim

    def forward(self,source_parse,ref_parse,ref_img):
        source_semantic_f = self.enc(source_parse)
        ref_semantic_f = self.enc(ref_parse)
        ref_warp_img, corr_ref2source = self.attn(source_semantic_f, ref_semantic_f, ref_img)
        return ref_warp_img

class CPMPlus(nn.Module):
    def __init__(self, args, parse_dim= 3, mx_channel= 32, lamA= 2, lamB= 2):
        super(CPMPlus, self).__init__()
        self.makeup = Makeup(args)
        self.scm = Semantic_Correspodance_Module(parse_dim + 2, mx_channel)
        self.args = args
        self.vgg = models.vgg16(pretrained=True)

        self.criterionL2 = torch.nn.MSELoss()

        self.lambda_A = lamA
        self.lambda_B = lamB

        if args.device == 'cpu':
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:0')

        self.freeze_layers()

    def freeze_layers(self):
        # Freeze VGG
        self.vgg.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

        # Freeze CPM backbone
        self.makeup.freeze_modules()

    def de_norm(self, x):
        out = (x + 1) / 2
        return out.clamp(0, 1)

    def vgg_forward(self, x):
        for i in range(18):
            x=self.vgg.features[i](x)
        return x

    def color_makeup(self, A_txt, B_txt, alpha):
        color_txt = self.makeup.makeup(A_txt, B_txt)
        color = self.makeup.render_texture(color_txt)
        color = self.makeup.blend_imgs(self.makeup.face, color * 255, alpha=alpha)
        return color
    
    def GANForward(self, imgA, imgB):
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        imgA = transform(imgA[0])
        imgB = transform(imgB[0])
        imgA = torch.Tensor(imgA[None, :, :, :]).to(self.device)
        imgB = torch.Tensor(imgB[None, :, :, :]).to(self.device)

        # Get makeup result
        fake_A, fake_B = self.makeup.color(imgA, imgB)
        result = self.de_norm(fake_A.detach()[0]).cpu().numpy().transpose(1, 2, 0)
        result = cv2.resize(result, (256, 256), cv2.INTER_CUBIC)
        return fake_A, fake_B

    def pattern_makeup(self, A_txt, B_txt, render_texture=False):
        mask = self.makeup.get_mask(B_txt)
        mask = (mask > 0.0001).astype("uint8")
        pattern_txt = A_txt * (1 - mask)[:, :, np.newaxis] + B_txt * mask[:, :, np.newaxis]
        pattern = self.makeup.render_texture(pattern_txt)
        pattern = self.makeup.blend_imgs(self.makeup.face, pattern, alpha=1)
        return pattern
    
    def criterionHis(self, input_data, target_data, mask_src, mask_tar, index):
        input_data = (self.de_norm(input_data) * 255).squeeze()
        target_data = (self.de_norm(target_data) * 255).squeeze()
        mask_src = mask_src.expand(1, 3, mask_src.size(2), mask_src.size(2)).squeeze()
        mask_tar = mask_tar.expand(1, 3, mask_tar.size(2), mask_tar.size(2)).squeeze()
        input_masked = input_data * mask_src
        target_masked = target_data * mask_tar
        # dstImg = (input_masked.data).cpu().clone()
        # refImg = (target_masked.data).cpu().clone()
        input_match = histogram_matching(input_masked, target_masked, index)
        input_match = self.to_var(input_match, requires_grad=False)
        loss = self.criterionL1(input_masked, input_match)
        return loss

    def uvTransform_img(self, imgA, imgB, np_format= False):
        self.makeup.prn_process(imgA)
        A_txt, A_pos = self.makeup.get_texture(), self.makeup.pos
        B_txt, B_pos = self.makeup.prn_process_target(imgB)

        tensorA = torch.Tensor(A_pos).to(self.device).unsqueeze(0)
        tensorA = torch.permute(tensorA, (0, 3, 1, 2))
        tensorB = torch.Tensor(B_pos).to(self.device).unsqueeze(0)
        tensorB = torch.permute(tensorB, (0, 3, 1, 2))
        tensorAtxt = torch.Tensor(A_txt).to(self.device).unsqueeze(0)
        tensorAtxt = torch.permute(tensorAtxt, (0, 3, 1, 2))
        tensorBtxt = torch.Tensor(B_txt).to(self.device).unsqueeze(0)
        tensorBtxt = torch.permute(tensorBtxt, (0, 3, 1, 2))
        
        # print(tensorA.shape, tensorB.shape, tensorimgB.shape)

        tensorBtxt = self.scm(tensorA, tensorB, tensorBtxt)
        
        if np_format:
          B_txt = tensorBtxt[0].detach().cpu().numpy()
          B_txt = (B_txt * 255).astype(np.uint8)
          B_txt = B_txt.transpose(1, 2, 0)
          return A_txt, A_pos, B_txt, B_pos

        return tensorAtxt, tensorA, tensorBtxt, tensorB
    
    def uvTransform(self, A_txt, A_pos, B_txt, B_pos):
        tensorA = torch.Tensor(A_pos).to(self.device)
        tensorA = torch.permute(tensorA, (0, 3, 1, 2))
        tensorB = torch.Tensor(B_pos).to(self.device)
        tensorB = torch.permute(tensorB, (0, 3, 1, 2))
        tensorAtxt = torch.Tensor(A_txt).to(self.device)
        tensorAtxt = torch.permute(tensorAtxt, (0, 3, 1, 2))
        tensorBtxt = torch.Tensor(B_txt).to(self.device)
        tensorBtxt = torch.permute(tensorBtxt, (0, 3, 1, 2))
        
        # print(tensorA.shape, tensorB.shape, tensorimgB.shape)

        tensorBtxt = self.scm(tensorA, tensorB, tensorBtxt)

        return tensorAtxt, tensorA, tensorBtxt, tensorB
    
    def calLoss(self, txtA, txtB):
        fake_A, fake_B = self.GANForward(txtA, txtB)
        fake_A = fake_A
        fake_B = fake_B

        # vgg loss
        vgg_org=self.vgg_forward(txtA)
        vgg_fake_A=self.vgg_forward(fake_A)
        g_loss_A_vgg = self.criterionL2(vgg_fake_A, vgg_org) * self.lambda_A

        vgg_ref=self.vgg_forward(txtB)
        vgg_fake_B=self.vgg_forward(fake_B)
        g_loss_B_vgg = self.criterionL2(vgg_fake_B, vgg_ref) * self.lambda_B

        vgg_loss = g_loss_A_vgg + g_loss_B_vgg
        # print(vgg_loss, vgg_fake_B)

        return vgg_loss
    
    def forward(self, imgA, imgB):
        A_txt, A_pos, B_txt, B_pos = self.uvTransform_img(imgA, imgB, np_format= True)
        # print(type(B_txt))
        # print(B_txt.shape)
        if self.args.color_only:
            output = self.color_makeup(A_txt, B_txt, self.args.alpha)
        elif self.args.pattern_only:
            output = self.pattern_makeup(A_txt, B_txt)
        else:
            color_txt = self.makeup.makeup(A_txt, B_txt) * 255
            mask = self.makeup.get_mask(B_txt)
            mask = (mask > 0.001).astype("uint8")
            new_txt = color_txt * (1 - mask)[:, :, np.newaxis] + B_txt * mask[:, :, np.newaxis]
            output = self.makeup.render_texture(new_txt)
            output = self.makeup.blend_imgs(self.makeup.face, output, alpha=1)

        x2, y2, x1, y1 = self.makeup.location_to_crop()
        output = np.concatenate([imgB[x2:], self.makeup.face[x2:], output[x2:]], axis=1)

        return A_txt, B_txt, output
            
def cal_hist(image):
    """
        cal cumulative hist for channel list
    """
    hists = []
    for i in range(0, 3):
        channel = image[i]
        # channel = image[i, :, :]
        channel = torch.from_numpy(channel)
        # hist, _ = np.histogram(channel, bins=256, range=(0,255))
        hist = torch.histc(channel, bins=256, min=0, max=256)
        hist = hist.numpy()
        # refHist=hist.view(256,1)
        sum = hist.sum()
        pdf = [v / sum for v in hist]
        for i in range(1, 256):
            pdf[i] = pdf[i - 1] + pdf[i]
        hists.append(pdf)
    return hists


def cal_trans(ref, adj):
    """
        calculate transfer function
        algorithm refering to wiki item: Histogram matching
    """
    table = list(range(0, 256))
    for i in list(range(1, 256)):
        for j in list(range(1, 256)):
            if ref[i] >= adj[j - 1] and ref[i] <= adj[j]:
                table[i] = j
                break
    table[255] = 255
    return table


def histogram_matching(dstImg, refImg, index):
    """
        perform histogram matching
        dstImg is transformed to have the same the histogram with refImg's
        index[0], index[1]: the index of pixels that need to be transformed in dstImg
        index[2], index[3]: the index of pixels that to compute histogram in refImg
    """
    index = [x.cpu().numpy() for x in index]
    dstImg = dstImg.detach().cpu().numpy()
    refImg = refImg.detach().cpu().numpy()
    dst_align = [dstImg[i, index[0], index[1]] for i in range(0, 3)]
    ref_align = [refImg[i, index[2], index[3]] for i in range(0, 3)]
    hist_ref = cal_hist(ref_align)
    hist_dst = cal_hist(dst_align)
    tables = [cal_trans(hist_dst[i], hist_ref[i]) for i in range(0, 3)]

    mid = copy.deepcopy(dst_align)
    for i in range(0, 3):
        for k in range(0, len(index[0])):
            dst_align[i][k] = tables[i][int(mid[i][k])]

    for i in range(0, 3):
        dstImg[i, index[0], index[1]] = dst_align[i]

    dstImg = torch.FloatTensor(dstImg).cuda()
    return dstImg