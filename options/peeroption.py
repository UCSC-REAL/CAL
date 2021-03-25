from .baseoption import BaseOptions


class PeerOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)

        # model: training-PeerLoss
        self._parser.add_argument('--is_peerloss', type=self.boolean_string, default=False, 
                                  help='whether to train with Peer Lossfunction')
        self._parser.add_argument('--peer_size', type=int, default=1, help="number of peer samples")
        self._parser.add_argument('--alpha', type=float, default=0.0, help='alpha in Peer Loss')
        self._parser.add_argument('--is_rescale_lr_by_alpha', type=self.boolean_string, default=True, 
                                  help="whether rescale the learning rate when peer term presents")

        # model: training-PeerLoss-Alpha_schduler
        self._parser.add_argument('--alpha_scheduler', type=str, default='seg', choices=['none','step','multistep','cosanneal','seg'],
                                  help="type of alpha scheduler")
        self._parser.add_argument('--alpha_step_size', type=int, default=20, help="step size for alpha decay")
        self._parser.add_argument('--milestones', nargs="+", type = int, default=None, help="list of milestones")
        self._parser.add_argument('--alpha_list', nargs="+", type = float, default=None, help="list of alphas")
        self._parser.add_argument('--gamma', type=float, default=0.1, help="alpha decay rate")
        self._parser.add_argument('--T_max', type=int, default=20, help="half period of Cos variation")
        self._parser.add_argument('--eta_min', type=float, default=0., help="Minimum alpha value")
