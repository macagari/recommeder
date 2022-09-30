
CF_MODELS = ['als', 'lmf', 'bpr']


class CFConfig:
    def __init__(self):
        self.factors = 20
        self.regularization = 0.01
        self.dtype = float
        self.iterations = 100
        self.random_state = 1234
        self.learning_rate = 0.01
        self.neg_prop = 10
