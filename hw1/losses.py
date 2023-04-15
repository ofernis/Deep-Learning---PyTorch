import abc
import torch


class ClassifierLoss(abc.ABC):
    """
    Represents a loss function of a classifier.
    """

    def __call__(self, *args, **kwargs):
        return self.loss(*args, **kwargs)

    @abc.abstractmethod
    def loss(self, *args, **kw):
        pass

    @abc.abstractmethod
    def grad(self):
        """
        :return: Gradient of the last calculated loss w.r.t. model
            parameters, as a Tensor of shape (D, C).
        """
        pass


class SVMHingeLoss(ClassifierLoss):
    def __init__(self, delta=1.0):
        self.delta = delta
        self.grad_ctx = {}

    def loss(self, x, y, x_scores, y_predicted):
        """
        Calculates the Hinge-loss for a batch of samples.

        :param x: Batch of samples in a Tensor of shape (N, D).
        :param y: Ground-truth labels for these samples: (N,)
        :param x_scores: The predicted class score for each sample: (N, C).
        :param y_predicted: The predicted class label for each sample: (N,).
        :return: The classification loss as a Tensor of shape (1,).
        """

        assert x_scores.shape[0] == y.shape[0]
        assert y.dim() == 1

        # TODO: Implement SVM loss calculation based on the hinge-loss formula.
        #  Notes:
        #  - Use only basic pytorch tensor operations, no external code.
        #  - Full credit will be given only for a fully vectorized
        #    implementation (zero explicit loops).
        #    Hint: Create a matrix M where M[i,j] is the margin-loss
        #    for sample i and class j (i.e. s_j - s_{y_i} + delta).

        loss = None
        # ====== LOTAN ATTEMPT: ======
        lotan = False
        if lotan:
            # ====== YOUR CODE: ======
            N = x_scores.shape[0]
            
            # # Create matrix M
            # ground_scores = x_scores[torch.arange(N), y].reshape(-1, 1)
            # M = x_scores - ground_scores + self.delta
            # M[M == self.delta] = float('-inf')
            
            # # Compute the hinge loss
            # L_i = torch.max(M, dim=1)[0]
            # L_i[L_i < 0] = 0
            # loss = torch.mean(L_i)
            # Calculate margin-loss matrix
            gt = x_scores[range(x_scores.shape[0]), y]
            gt = torch.reshape(gt, [gt.shape[0], 1])
            ground_scores = x_scores.gather(1, y.view(-1, 1))
            M = x_scores - ground_scores + self.delta
            # Zero out the margin loss for the correct class
            M[M == self.delta] = 0
            # Calculate the loss for each sample
            loss_per_sample, _ = torch.max(M, dim=1)
            # Compute the average loss over all samples
            loss = torch.mean(loss_per_sample)
            
            # TODO: Save what you need for gradient calculation in self.grad_ctx
            # ====== YOUR CODE: ======
            self.grad_ctx['X'] = x
            self.grad_ctx['M'] = M
            self.grad_ctx['y'] = y
            # ========================
            
            return loss
        # ========================
        # ====== YOUR CODE: ======
        ground_scores = x_scores[range(x_scores.shape[0]), y]
        ground_scores = torch.reshape(ground_scores, [ground_scores.shape[0], 1])
        M = x_scores - ground_scores + self.delta
        M[M == self.delta] = 0
        M[M < 0] = 0
       
        N = x_scores.shape[0]
        loss = (1 / N) * torch.sum(M)
        # ========================

        # TODO: Save what you need for gradient calculation in self.grad_ctx
        # ====== YOUR CODE: ======
        self.grad_ctx['X'] = x
        self.grad_ctx['M'] = M
        self.grad_ctx['y'] = y
        # ========================

        return loss

    def grad(self):
        """
        Calculates the gradient of the Hinge-loss w.r.t. parameters.
        :return: The gradient, of shape (D, C).

        """
        # TODO:
        #  Implement SVM loss gradient calculation
        #  Same notes as above. Hint: Use the matrix M from above, based on
        #  it create a matrix G such that X^T * G is the gradient.

        grad = None
        # ====== LOTAN ATTEMPT: ======
        lotan = False
        if lotan:
            # ====== YOUR CODE: ======
            x = self.grad_ctx['X']
            M = self.grad_ctx['M']
            y = self.grad_ctx['y']
            N, C = M.shape

            # Compute the gradient
            G = torch.zeros((N,C))
            G[M > 0] = 1
            G[torch.arange(N), y] -= torch.sum(G, dim=1)
            grad = torch.matmul(x.T, G) / N

            return grad 
        # ========================
        # ====== YOUR CODE: ======
        X = self.grad_ctx['X']
        M = self.grad_ctx['M']
        y = self.grad_ctx['y']
        
        G = (M > 0).float()
        N = G.shape[0]       
        non_zero_mask = (M != 0)

        row_counts = -1 * torch.sum(non_zero_mask, dim=1)
        G[range(N), y] = row_counts.float()
        
        grad = X.T @ G / N
        # ========================

        return grad 
