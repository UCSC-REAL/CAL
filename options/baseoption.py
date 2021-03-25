
import argparse
import os

class BaseOptions:
    def __init__(self):
        self._parser = argparse.ArgumentParser()
        self._initialized = False

    def initialize(self):
        # control
        self._parser.add_argument('--is_train', type=self.boolean_string, default=True, 
                                  help='whether we train or validate the model')
        self._parser.add_argument('--is_load', type=self.boolean_string, default=False, help='whether load existing model')
        self._parser.add_argument('--is_continue_train', type=self.boolean_string, default=False, help='whether continue training')
        self._parser.add_argument('--is_plot_results', type=self.boolean_string, default=False, 
                                  help='whether plot loss and acc figures after training for each epoch')
        self._parser.add_argument('--is_class_resolved', type=self.boolean_string, default=False, 
                                  help='whether plot cluster-resolved distributions')
        self._parser.add_argument('--gpu_idx', type=str, default="0", help="gpu indices used for training")

        # model: general
        self._parser.add_argument('--exp_name', required=True, type=str, default='test_code', 
                                  help='a unique name for experiment')
        self._parser.add_argument('--log_root', type=str, default="./logs", 
                                  help='root folder for recording experiment results')
        self._parser.add_argument('--netARCH', type=str, default='lenet', help='network architecture')
        self._parser.add_argument('--is_check_loss_dist', type=self.boolean_string, default=True, 
                                  help='whether to compute and record loss distribution during training')

        # model: training-general
        self._parser.add_argument('--max_epoch', type=int, default=10, 
                                  help='maximum number of epochs for training the model')
        self._parser.add_argument('--batch_size', type=int, default=64, help='batch size')
        self._parser.add_argument('--num_workers', type=int, default=1, help='number of works in Data Loader')

        # model: training-optimizer & learning rate
        self._parser.add_argument('--optimizer', type=str, default='adam', help='optimizer')
        self._parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
        self._parser.add_argument('--weight_decay', type=float, default=0.0, help='SGD weight_decay')

        self._parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
        self._parser.add_argument('--lr_scheduler', type=str, choices=['none','step','multistep','SGDR'], 
                                  default='step', help="type of learning rate scheduler")
        self._parser.add_argument('--lr_decay_step_size', type=int, default=30, help='step size for lr decay')
        self._parser.add_argument('--lr_decay_rate', type=float, default= 0.1, help='rate for lr decay')
        self._parser.add_argument('--lr_milestones', nargs="+", type = int, default=None, help="list of milestones for lr")
        self._parser.add_argument('--lr_gamma', type=float, default=0.1, help="alpha decay rate of lr")
        self._parser.add_argument('--lr_eta_min', type=float, default=0.0, help="lower val of cosine lr")
        self._parser.add_argument('--lr_T_max', type=int, default=0, help="half period of cosine variation for lr")
        self._parser.add_argument('--lr_T', type=int, default=60, help="period for lr to restart")

        # model: training-lossfunction (including Bi Tempered loss)
        self._parser.add_argument('--lossfunc', type=str, default='crossentropy', help='lossfunction')
        self._parser.add_argument('--T1', type=float, default=0.57, help='temperature 1; 0<t1<1')
        self._parser.add_argument('--T2', type=float, default=2.76, help='temperature 2; 1<t2')

        # model: validate
        self._parser.add_argument('--is_validate', type=self.boolean_string, default=True, 
                                  help='whether to use part of the training data as validation')
        self._parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio of the training data as val')

        # dataset: general
        self._parser.add_argument('--data_root', type=str, default='./datasets',
                                  help='the root path for datasets')
        self._parser.add_argument('--dataset', type=str, default='MNIST', help='dataset')
        self._parser.add_argument('--num_classes', type=int, default=None, help='number of classes')
        self._parser.add_argument('--chosen_classes', type=int, nargs="+", default=None, help='choose binary classes')
        self._parser.add_argument('--spl_ckpt', type=int, nargs="+", default=[300], help='check point for SPL')
        self._parser.add_argument('--sample_weight_path', type=str, default=None, help='the path of sample weight')
        self._parser.add_argument('--beta_path', type=str, default=None, help='beta for distilled samples')
        self._parser.add_argument('--ratio_true', type=float, default=1, help='ratio of true labels (for distilled BILN only)')
        self._parser.add_argument('--model_path', type=str, default=None, help='load model')
        
        # dataset: noise
        self._parser.add_argument('--with_noise', type=self.boolean_string, default=False, 
                                  help='whether the training labels are noisy')
        self._parser.add_argument('--noise_label_fname', type=str, default='noise_label_train.pt', 
                                  help='specify the noise label filename')

        self._initialized = True

    def boolean_string(self, s):
        if s not in {'False', 'True', '0', '1'}:
            raise ValueError('Not a valid boolean string')

        return (s == 'True') or (s == '1')

    def parse(self):
        if not self._initialized:
            self.initialize()

        opt = self._parser.parse_args()

        return opt

    def parse_args(self, param_dict):
        if not self._initialized:
            self.initialize()

        opt = self._parser.parse_args(param_dict)

        return opt