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
        pred_probabilities = F.softmax(prediction, dim=1).data.squeeze()
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


class Paul_CAM():
    def __init__(self,model,device,last_conv_layer='layer3',fc_layer_name='fc1'):
        
        self.device = device
        self.last_conv_layer = last_conv_layer
        self.fc_layer_name = fc_layer_name
        self.model = model


    def run(self,instance,label_instance=None):
        cam,label_pred =  self.__get_CAM_class(np.array(instance))
        if (label_instance is not None) and (label_pred != label_instance):
            #Verbose
            print("Expected classification as class {} but got class {}".format(label_instance,label_pred))
            print("The Class activation map is for class {}".format(label_instance,label_pred))
        return cam

    

    # ================Private methods=====================   

    def __getCAM(self,feature_conv, weight_fc, class_idx):
        _, nc, length = feature_conv.shape
        feature_conv_new = feature_conv
        cam = weight_fc[class_idx].dot(feature_conv_new.reshape((nc,length)))
        cam = cam.reshape(length)
        
        return cam


    def __get_CAM_class(self,instance):
        original_dim = len(instance)
        original_length = len(instance[0])
        instance_to_try = Variable(
            torch.tensor(
                instance.reshape(
                    (1,original_dim,original_length))).float().to(self.device),
            requires_grad=True)
        final_layer = self.last_conv_layer
        activated_features = SaveFeatures(final_layer)
        prediction = self.model(instance_to_try)
        pred_probabilities = F.softmax(prediction).data.squeeze()
        activated_features.remove()
        weight_softmax_params = list(self.fc_layer_name.parameters())
        weight_softmax = np.squeeze(weight_softmax_params[0].cpu().data.numpy())
        
        class_idx = topk(pred_probabilities,1)[1].int()
        overlay = self.__getCAM(activated_features.features, weight_softmax, class_idx )
        
        return overlay,class_idx.item()
