import numpy as np
from torch.autograd import Variable
import torch
from torch.nn import functional as F
from torch import topk


class SaveFeatures():
    features=None
    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output): self.features = ((output.cpu()).data).numpy()
    def remove(self): self.hook.remove()


def fmax(val):
    if val > 0 : return val
    else: return 0
def fmin(val):
    if val < 0 : return val
    else: return 0
    

class CAM():
    def __init__(self, model, device, last_conv_layer='layer3', fc_layer_name='fc1', verbose=True):
        
        self.device = device
        self.last_conv_layer = last_conv_layer
        self.fc_layer_name = fc_layer_name
        self.model = model
        self.verbose = verbose


    def run(self,instance, label_instance=None, returned_cam_for_label=None):
        assert label_instance is not None or returned_cam_for_label is not None
        cam, label_pred =  self.__get_CAM_class(np.array(instance), returned_cam_for_label)

        return cam, label_pred

    # ================Private methods=====================   

    def __getCAM(self,feature_conv, weight_fc, class_idx):
        _, nc, length = feature_conv.shape
        feature_conv_new = feature_conv
        cam = weight_fc[class_idx].dot(feature_conv_new.reshape((nc,length)))
        cam = cam.reshape(length)
        
        return cam


    def __get_CAM_class(self, instance, returned_cam_for_label):
        assert len(instance.shape)==3, 'Please provide batched Tensor data (3 dims).'

        instance_to_try = Variable(torch.tensor(instance).float().to(self.device), requires_grad=True)

        final_layer = self.last_conv_layer
        activated_features = SaveFeatures(final_layer)
        prediction = self.model(instance_to_try)
        pred_probabilities = F.softmax(prediction, dim=-1).data.squeeze()
        activated_features.remove()
        weight_softmax_params = list(self.fc_layer_name.parameters())
        weight_softmax = np.squeeze(weight_softmax_params[0].cpu().data.numpy())
        
        class_idx = topk(pred_probabilities, 1)[1].int()
        
        if self.verbose:
            print('Pred proba :', pred_probabilities.cpu().numpy())

        if returned_cam_for_label is None:
            overlay = self.__getCAM(activated_features.features, weight_softmax, class_idx)
        else:
            overlay = self.__getCAM(activated_features.features, weight_softmax, torch.Tensor(np.array([returned_cam_for_label])).int().to(self.device))
        
        return overlay, class_idx.item()


class AttentionMap():
    def __init__(self, model, device, n_encoder_layers=1, merge_channels_att='sum', head_att='mean'):
        
        self.device = device
        self.model = model
        self.merge_channels_att = merge_channels_att
        self.head_att = head_att
        self.n_encoder_layers = n_encoder_layers

    def run(self, instance, return_att_for='all'):
        self.model.eval()
        
        pred = self.model(torch.Tensor(instance).to(self.device))
        pred = torch.nn.Softmax(dim=-1)(pred).detach().cpu().numpy()[0]
        
        if return_att_for=='all':
            att = []
            for n_e in range(self.n_encoder_layers):
                att.append(self._extract_att_one_block(n_e))
            att = np.array(att).mean(axis=0)
        else:
            assert return_att_for < self.n_encoder_layers
            att = self._extract_att_one_block(return_att_for)
        
        return att, pred
    
    def _extract_att_one_block(self, i):
        att = self.model._modules['EncoderBlock'][i].att.detach().cpu().numpy()[0]

        if self.merge_channels_att=='sum':
            att = att.sum(axis=1)
        else:
            att = att.mean(axis=1)

        if self.head_att=='sum':
            att = att.sum(axis=0)
        elif self.head_att=='mean':
            att = att.mean(axis=0)
        else:
            att = att[i, :]
            
        return att
