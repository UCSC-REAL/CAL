import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ =[
    "CrossEntropyLossStable", "CrossEntropyLossRegStable", "CrossEntropyStableCALMultiple", "PeerLossRegCE"
]




# ======================================================================== #
# Peer loss with fixed peer sample size = 1
# ======================================================================== #
class CrossEntropyLossStable(nn.Module):
    '''
    For use in PeerLossOne, as the original CrossEntropyLoss are likely be
    blowup when using in the peer term
    '''
    def __init__(self, reduction='mean', eps=1e-8):
        super(CrossEntropyLossStable, self).__init__()
        self._name = "Stable Cross Entropy Loss"
        self._eps = eps
        self._softmax = nn.Softmax(dim=-1)
        self._nllloss = nn.NLLLoss(reduction=reduction)

    def forward(self, outputs, labels):
        return self._nllloss( torch.log( self._softmax(outputs) + self._eps ), labels)



class CrossEntropyLossRegStable(nn.Module):
    def __init__(self, noisy_prior, eps=1e-5): 
        # eps = 1e-5 for most settings
        # try to find the eps for mixup settings
        super(CrossEntropyLossRegStable, self).__init__()
        self._prior = noisy_prior
        self._eps = eps
        self._softmax = nn.Softmax(dim=-1)
    
    def forward(self, outputs, clusteridx = None):
        r'''
        labels are real vectors (from one-hot-encoding)
        ------
        outputs : batch * num_class
        labels  : batch * num_class
        '''
        log_out = torch.log( self._softmax(outputs) + self._eps )
        noise_prior = torch.zeros_like(outputs)
        res = torch.sum(torch.mul(self._prior, log_out), dim=1)
        return -torch.mean(res)


class CrossEntropyStableCALMultiple(nn.Module):
    def __init__(self, T_mat, T_mat_true,P_y_distill, eps=1e-5): 
        # eps = 1e-5 for most settings
        # try to find the eps for mixup settings
        super(CrossEntropyStableCALMultiple, self).__init__()
        self.T_mat = T_mat
        self.T_mat_true = T_mat_true
        self._eps = eps
        self._softmax = nn.Softmax(dim=-1)
        self.P_y_distill = P_y_distill
    
    def forward(self, outputs, true_y, distill_y, raw_idx, loss_mean_y, loss_mean_n, loss_mean_all, loss_mean_y_true, loss_mean_n_true, loss_mean_all_true, distilled_weights):
        r'''
        labels are real vectors (from one-hot-encoding)
        ------
        outputs : batch * num_class
        labels  : batch * num_class
        true_y is actually the distilled y
        distilled_weights is beta
        '''
        log_out = -torch.log( self._softmax(outputs) + self._eps )
        T_mat = self.T_mat[:,:,raw_idx]
        T_mat_indicator = torch.sum(T_mat>0.0,1).view(T_mat.shape[0],1,-1).repeat(1,T_mat.shape[0],1).float()
        if self.T_mat_true is not None:
            T_mat_true = self.T_mat_true[:,:,raw_idx]
            T_mat_indicator_true = torch.sum(T_mat_true>0.0,1).view(T_mat_true.shape[0],1,-1).repeat(1,T_mat_true.shape[0],1).float()
        loss_all = torch.transpose(log_out,0,1).view(1,T_mat.shape[0],-1).repeat(T_mat.shape[0],1,1)
        # (torch.sum(T_mat_indicator * loss_all, dim = 2)/torch.sum(T_mat_indicator, dim = 2))
        loss_all_norm = (loss_all - loss_mean_all.view(T_mat.shape[0],T_mat.shape[0],1).repeat(1,1,T_mat.shape[2])) * T_mat_indicator
        if self.T_mat_true is not None:
            loss_all_norm_true = (loss_all - loss_mean_all.view(T_mat.shape[0],T_mat.shape[0],1).repeat(1,1,T_mat.shape[2])) * T_mat_indicator_true
            loss_rec_all_true = torch.sum(T_mat_indicator_true * loss_all, dim = 2)
        loss_rec_all = torch.sum(T_mat_indicator * loss_all, dim = 2)
        

        T_mat_indicator_sum = torch.sum(T_mat_indicator, dim = 2)
        T_mat_indicator_sum[T_mat_indicator_sum==0] = 1.0
        if self.T_mat_true is not None:
            T_mat_indicator_true_sum = torch.sum(T_mat_indicator_true, dim = 2)
            T_mat_indicator_true_sum[T_mat_indicator_true_sum==0] = 1.0
            loss_out_true = torch.sum(torch.sum(torch.sum(T_mat_true * loss_all_norm_true, dim = 2)/T_mat_indicator_true_sum, dim = 1) * 1.0/T_mat.shape[0])
        loss_out = torch.sum(torch.sum(torch.sum(T_mat * loss_all_norm, dim = 2)/T_mat_indicator_sum, dim = 1) * self.P_y_distill) # TODO: test even p_y_distill
        if self.T_mat_true is not None:
            return loss_out, loss_out_true, loss_rec_all, loss_rec_all_true
        else:
            return loss_out, torch.tensor(0.0), loss_rec_all, torch.tensor(0.0)



class CrossEntropyLossRegStableMix(nn.Module):
    def __init__(self, noisy_prior, eps=1e-2):  # 
        # eps = 1e-5 for most settings
        # try to find the eps for mixup settings
        super(CrossEntropyLossRegStableMix, self).__init__()
        # noisy_prior_0 = noisy_prior.repeat(noisy_prior.shape[0],1)
        # noisy_prior_1 = noisy_prior_0.transpose(0,1)
        # self._prior = noisy_prior_0 + noisy_prior_1
        self._prior = noisy_prior
        self._eps = eps
        self._softmax = nn.Softmax(dim=-1)
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")
    
    def forward(self, outputs, lmd, noise_prior_new = None, weight = None):
        r'''
        labels are real vectors (from one-hot-encoding)
        ------
        outputs : batch * num_class    mixup outputs
        labels  : batch * num_class
        y1, y2 are noisy labels
        \sum_{y1,y2} P(y1 or y2)*log(f_x(y1)+f_x(y2))
        '''   
        length = outputs.shape[0]//2
        # log_out = torch.log( torch.abs(self._softmax(outputs[:length]) - lmd) + self._eps )
        log_out = torch.log( self._softmax(outputs[length:]) + self._eps )
        if noise_prior_new is not None:
            res = torch.sum(torch.mul(noise_prior_new, log_out), dim=1)
        else:
            res = torch.sum(torch.mul(self._prior, log_out), dim=1)
        if weight is not None:
            weight_var = torch.tensor(weight).to(self._device)
            if sum(weight) == 0:
                return -torch.sum(res*weight_var)
            else:
                return -torch.sum(res*weight_var)/sum(weight)
        else:
            return -torch.mean(res)


        return -torch.mean(res)

class PeerLossRegCE(nn.Module):
    def __init__(self, alpha, noisy_prior, loss_name, T_mat = None, T_mat_true = None, P_y_distill = None):
        super(PeerLossRegCE, self).__init__()
        self._name = "peer loss function with noisy prior"
        self._lossname = loss_name
        if loss_name == 'crossentropy':
            self._peer = CrossEntropyLossRegStable(noisy_prior)
        elif loss_name == 'crossentropy_CAL':
            self._peer = CrossEntropyLossRegStable(noisy_prior)
            self._CAL = CrossEntropyStableCALMultiple(T_mat,T_mat_true,P_y_distill)

        self._ce = CrossEntropyLossStable()
        self._alpha = alpha if alpha is not None else 1.
    
    def forward(self, outputs, labels, output_peer, labels_nomix = None, lmd = None, noisy_prior_new = None, weight = None, true_y = None, distill_y = None, raw_idx = None, loss_mean_y = None, loss_mean_n = None, loss_mean_all = None, distilled_weights = None, loss_mean_y_true = None, loss_mean_n_true = None, loss_mean_all_true = None):
        # calculate the base loss
        base_loss = self._ce(outputs, labels)
        peer_term = self._peer(output_peer)
        if self._lossname == "crossentropy_CAL":
            CAL_term, CAL_term_true, loss_rec_all, loss_rec_all_true = self._CAL(outputs, true_y, distill_y, raw_idx, loss_mean_y, loss_mean_n, loss_mean_all, loss_mean_y_true, loss_mean_n_true, loss_mean_all_true, distilled_weights)
            return base_loss - self._alpha * peer_term - CAL_term, base_loss, peer_term, CAL_term.detach(), loss_rec_all.detach()
        else:
            return base_loss - self._alpha * peer_term, base_loss, peer_term
# ======================================================================== #




if __name__ == "__main__":
    pass