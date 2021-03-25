from .peeroption import PeerOptions


class PeerRegOptions(PeerOptions):
    def initialize(self):
        PeerOptions.initialize(self)
        self._parser.add_argument('--nosiy_prior', nargs="+", type = float, default=None, help="list of noisy class dist")