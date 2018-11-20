import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torchvision
import numpy as np
import torch

W_IMG = 224
H_IMG = 224 # 224
N_ANCHORS = 3

class VGG_RP(nn.Module):

    def __init__(self, features, num_classes=20,init_weights=False):
        super(VGG_RP, self).__init__()
        self.features = features
        self.classifier2 = nn.Sequential(
            nn.Linear(512 * 7 * 7, num_classes),
            # nn.ReLU(True),
            # nn.Dropout(),
            # nn.Linear(50, num_classes),
        )

        # stop gradients ? which layers ?
        # for parameter in model.parameters():
        #     parameter.requires_grad = False

        if init_weights:
            self._initialize_weights(self.modules())
        else:
            self._initialize_weights(self.classifier2.children())

        self.region_proposal_window = nn.Sequential(*[
            nn.Conv2d(512, 512, kernel_size=3, padding=0),
            nn.ReLU(True)
            ])
        self.region_proposal_cls = nn.Sequential(*[
            nn.Conv2d(512, N_ANCHORS, kernel_size=1, padding=0),
            nn.Sigmoid()
            ])
        self.region_proposal_reg = nn.Conv2d(512, 4*N_ANCHORS, kernel_size=1, padding=0)

        self._initialize_rp_weights()
        self._create_anchors()

    def forward(self, x): # 224 x 224 x N_ANCHORS
        x = x.float()
        x = self.features(x) # 7 x 7 x 512

        rp_512 = self.region_proposal_window(x) # 5 x 5 x 512
        rp_cls = self.region_proposal_cls(rp_512) # 5 x 5 x N_ANCHORS
        rp_reg = self.region_proposal_reg(rp_512) # 5 x 5 x 12
        rp_cls = rp_cls.view(x.size(0),5,5,3) # 5 x 5 x N_ANCHORS x 4 # tx,ty,tw,th
        rp_reg = rp_reg.view(x.size(0),5,5,3,4) # 5 x 5 x N_ANCHORS x 4 # tx,ty,tw,th

        self._compute_bboxes(rp_cls,rp_reg)

        x = x.view(x.size(0), -1)
        x = self.classifier2(x)
        return x, rp_cls, rp_reg

    def _initialize_weights(self,layers):
        for m in layers:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _initialize_rp_weights(self):
        for m in self.region_proposal_window.children():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        for m in self.region_proposal_cls.children():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        for m in self.region_proposal_reg.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _create_anchors(self):
        scales = [56,112,224]
        anchors = np.zeros((int(W_IMG/32)-2,int(H_IMG/32)-2,N_ANCHORS,8)) # 5 x 5 x N_ANCHORS x 8
        for x in range(anchors.shape[0]):
            for y in range(anchors.shape[1]):
                for anchor in range(anchors.shape[2]):
                    w,h = scales[anchor], scales[anchor]
                    cx,cy = x,y
                    x0,y0 = x-int(w/2), y-int(h/2)
                    # x0,y0 = max(x0,0), max(y0,0)
                    x1,y1 = x+int(w/2), y+int(h/2)
                    # x1,y1 = min(x1,W_IMG-1), min(y1,H_IMG-1)
                    anchors[x,y,anchor,:] = [x0,y0,x1,y1,x,y,w,h] # 8 coord
        self.anchors = torch.tensor(anchors,requires_grad=False).cuda().float()

    def _compute_bboxes(self,rp_cls,rp_reg):
        """
        De-parameterize rp_reg
        Perform NMS
        """
        batch_size = rp_reg.size(0)
        anchors = self.anchors.repeat(batch_size,1,1,1,1)

        self.bboxes = torch.stack([
            rp_reg[:,:,:,:,0]*anchors[:,:,:,:,6]+anchors[:,:,:,:,4],
            rp_reg[:,:,:,:,1]*anchors[:,:,:,:,7]+anchors[:,:,:,:,5],
            torch.exp(rp_reg[:,:,:,:,2])*anchors[:,:,:,:,6],
            torch.exp(rp_reg[:,:,:,:,3])*anchors[:,:,:,:,7]
            ],dim=4) # format cx,cy,w,h
        self.bboxes = torch.stack([
            self.bboxes[:,:,:,:,0]-self.bboxes[:,:,:,:,2]/2,
            self.bboxes[:,:,:,:,1]-self.bboxes[:,:,:,:,3]/2,
            self.bboxes[:,:,:,:,0]+self.bboxes[:,:,:,:,2]/2,
            self.bboxes[:,:,:,:,1]+self.bboxes[:,:,:,:,3]/2
            ],dim=4).float().cuda() # format x0,y0,x1,y1


        sorted_bboxes = []
        for _ in range(batch_size):
            sorted_bboxes.append([])
            for x in range(anchors.shape[1]):
                for y in range(anchors.shape[2]):
                    for anchor in range(anchors.shape[3]):
                        sorted_bboxes[-1].append([x,y,anchor,rp_cls[_,x,y,anchor].item()])
            sorted_bboxes[-1] = np.array(sorted_bboxes[-1]) # TODO this is not clean .. we both need list and numpy structures 
            sorted_bboxes[-1] = sorted_bboxes[-1][np.argsort(-sorted_bboxes[-1][:][3])] # classification by confidence of the bboxes (descending)
            sorted_bboxes[-1] = sorted_bboxes[-1].tolist()

        # perform NMS
        for _ in range(batch_size):
            ind = 0
            while ind<len(sorted_bboxes[_])-1:
                ind2 = ind+1
                bbox_indexes = sorted_bboxes[_][ind][:3]
                bbox = self.bboxes[_,int(bbox_indexes[0]),int(bbox_indexes[1]),int(bbox_indexes[2])]
                while ind2<len(sorted_bboxes[_]):
                    bbox_indexes2 = sorted_bboxes[_][ind2][:3]
                    bbox2 = self.bboxes[_,int(bbox_indexes2[0]),int(bbox_indexes2[1]),int(bbox_indexes2[2])]
                    if bb_intersection_over_union(bbox, bbox2) > 0.7:
                        sorted_bboxes[_].pop(ind2)
                    else:
                        ind2 += 1
                ind += 1
        
        self.sorted_bboxes = sorted_bboxes


    def compute_rp_loss(self,bbox_reals,rp_cls,rp_reg,lamb=3):
        """
        Compute IOUs to get "pos" and "neg" anchors
        Sample 1:1 pos:neg (up to 256 each)
        Compute reg and cls losses
        """
        
        anchors = self.anchors
        bboxes = self.bboxes
        sorted_bboxes = self.sorted_bboxes
        batch_size = rp_reg.size(0)

        neg_anchors = []
        pos_anchors = []
        t_stars = []

        for _ in range(batch_size):
            neg_anchors.append([])
            pos_anchors.append([])
            t_stars.append([])
            for bbox_info in sorted_bboxes[_]:
                bbox = bboxes[_,int(bbox_info[0]),int(bbox_info[1]),int(bbox_info[2])]
                iou = bb_intersection_over_union(bbox.cuda().float(),bbox_reals[_].cuda().float())
                if iou > 0.7:
                    pos_anchors[-1].append([int(bbox_info[0]),int(bbox_info[1]),int(bbox_info[2])])
                    t_stars[-1].append(torch.tensor([
                        (bbox_reals[_][4].cuda().float()-anchors[int(bbox_info[0]),int(bbox_info[1]),int(bbox_info[2]),4])/anchors[int(bbox_info[0]),int(bbox_info[1]),int(bbox_info[2]),6],
                        (bbox_reals[_][5].cuda().float()-anchors[int(bbox_info[0]),int(bbox_info[1]),int(bbox_info[2]),5])/anchors[int(bbox_info[0]),int(bbox_info[1]),int(bbox_info[2]),7],
                        torch.log(bbox_reals[_][6].cuda().float()/anchors[int(bbox_info[0]),int(bbox_info[1]),int(bbox_info[2]),6]),
                        torch.log(bbox_reals[_][7].cuda().float()/anchors[int(bbox_info[0]),int(bbox_info[1]),int(bbox_info[2]),7])
                        ]).cuda().float())
                elif iou < 0.3:
                    neg_anchors[-1].append([int(bbox_info[0]),int(bbox_info[1]),int(bbox_info[2])])

        # nb_samples = min(len(neg_anchors),len(pos_anchors),256)

        loss_reg = torch.tensor(0).cuda().float()
        loss_cls = torch.tensor(0).cuda().float()

        for _ in range(batch_size):
            for neg in neg_anchors[_]:
                loss_cls += - torch.log(1-rp_cls[_,neg[0],neg[1],neg[2]])/len(neg_anchors[_]) # normalize by number of samples in this batch size
            for pos, t_star in zip(pos_anchors[_], t_stars[_]):
                loss_cls += - torch.log(rp_cls[_,pos[0],pos[1],pos[2]])
                loss_reg += torch.nn.SmoothL1Loss()(rp_reg[_,pos[0],pos[1],pos[2],:], t_star) # also normalize by number of samples in this batch size

        loss_reg *= lamb # already normalized by number of samples by torch.nn.SmootL1Loss

        return loss_cls, loss_reg


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
 
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / boxAArea + boxBArea - interArea
 
    # return the intersection over union value
    return iou




def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg16_bn_RP(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG_RP(make_layers(cfg['D'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(torchvision.models.vgg16_bn(pretrained=True).state_dict(),strict=False)
    return model


