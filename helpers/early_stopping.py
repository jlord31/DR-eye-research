class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, mode='max', verbose=True):
        """
        Early stopping to stop training when validation metric doesn't improve.

        Args:
            patience (int): How many epochs to wait after last improvement
            min_delta (float): Minimum change to qualify as an improvement
            mode (str): 'min' for loss, 'max' for accuracy
            verbose (bool): Whether to print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.verbose = verbose
        self.mode = mode

        self.delta = -min_delta if mode == 'max' else min_delta

    def __call__(self, metric):
        score = metric if self.mode == 'max' else -metric

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0