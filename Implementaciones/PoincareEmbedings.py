from math import dist
import numpy as np
import random

class PoincareEmbedding:
    def __init__(self, dim: int, STABILITY = 1e-3) -> None:
        """
        Initializes the PoincareEmbedding class.

        Args:
            dim (int): The dimension of the embeddings.
        """
        self.dim = dim
        self.STABILITY = STABILITY
        self.theta = dict()
        self.vocab = None
    
    def distance(self, u:np.ndarray, v:np.ndarray) -> float:
        """
        Computes the Poincaré distance between two embeddings u and v.

        Args:
            u: Embedding vector for the first entity.
            v: Embedding vector for the second entity.

        Returns:
            float: The Poincaré distance between u and v.
        """
        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)
        euclidean_dist = np.linalg.norm(u - v)
        alpha = 1 - norm_u ** 2
        beta = 1 - norm_v ** 2
        gamma = 1 + 2 * (euclidean_dist ** 2) / (alpha * beta)
        return np.arccosh(gamma)
     
    def partial_d(self, u:np.ndarray, v:np.ndarray) -> float:
        """
        Computes a partial derivative for optimization.

        Args:
            u: Embedding vector for the first entity.
            v: Embedding vector for the second entity.

        Returns:
            float: A partial derivative value.
        """
        alpha = 1 - np.linalg.norm(u) ** 2
        beta = 1 - np.linalg.norm(v) ** 2
        gamma = 1 + 2 * ((np.linalg.norm(u - v) ** 2) / (alpha * beta))
        return 4 * ((np.linalg.norm(v) ** 2 - 2 * np.dot(u, v) + 1) / alpha ** 2 * u - v/alpha )/ np.sqrt(gamma ** 2 - 1)
    
    def generate_negative_examples(self, data: dict, pos1, num_negatives = 5):
        """
        Generates negative examples for training.

        Args:
            data (dict): A dictionary representing the training data.
            pos1: The first entity in a positive example.
            num_negatives (int): The number of negative examples to generate.

        Returns:
            negs: A list of negative example pairs.
            dist_negs: A list of corresponding distances.
        """
        negs = []
        dist_negs_init = []
        dist_negs = []
        while (len(negs) < num_negatives):
            neg1 = pos1
            neg2 = random.choice(self.vocab)
            if not (neg2 in data[neg1] or neg1 in data[neg2] or neg2 == neg1):
                dist_neg_init = dist(self.theta[neg1], self.theta[neg2])
                negs.append([neg1, neg2])
                dist_neg = np.cosh(dist_neg_init)
                dist_negs_init.append(dist_neg_init)
                dist_negs.append(dist_neg)
        return np.array(negs), np.array(dist_negs)

                
    def calculate_denominator(self, negative_distances: np.ndarray) -> float:
        """
        Calculates the loss based on negative examples.

        Args:
            negative_distances (list): A list of negative distances.

        Returns:
            float: The calculated loss.
        """
        return np.sum(np.exp(-1*negative_distances))
    
    def gradient(self, negative_distances: np.ndarray, negatives: np.ndarray, pos1, pos2, loss_denominator: float):
        """
        Computes gradients for optimization.

        Args:
            negative_distances (list): A list of negative distances.
            negatives (list): A list of negative examples.
            pos1: The first entity in a positive example.
            pos2: The second entity in a positive example.
            loss_denominator (float): The loss denominator.

        Returns:
            Tuple: Gradients for positive and negative examples.
        """
        # Derivative of loss wrt positive relation [d(u, v)]
        der_p = -1
        der_negs = []
        # Derivative of loss wrt negative relation [d(u, v')]
        der_negs = np.exp(-1 * negative_distances) / (loss_denominator + self.STABILITY)
        # Derivative of loss wrt pos1
        der_p_pos1 = der_p * self.partial_d(self.theta[pos1], self.theta[pos2])
        # Derivative of loss wrt pos2
        der_p_pos2 = der_p * self.partial_d(self.theta[pos2], self.theta[pos1])
        der_negs_final = []
        for (der_neg, neg) in zip(der_negs, negatives):
            # derivative of loss wrt second element of the pair in neg
            der_neg1 = der_neg * self.partial_d(self.theta[neg[1]], self.theta[neg[0]])
            # derivative of loss wrt first element of the pair in neg
            der_neg0 = der_neg * self.partial_d(self.theta[neg[0]], self.theta[neg[1]])
            der_negs_final.append([der_neg0, der_neg1])
        return der_p_pos1, der_p_pos2, der_negs_final

    def update_embedding(self, theta, grad, lr):
        """
        Updates an embedding using a gradient and learning rate.

        Args:
            theta: The embedding to update.
            grad: The gradient for the embedding.
            lr: The learning rate.

        Returns:
            Updated embedding.
        """
        try:
            update =  lr*(1 - np.dot(theta,theta)**2)**2/4
            theta = theta - update * grad
            if (np.dot(theta, theta) >= 1):
                theta = theta/np.sqrt(np.dot(theta, theta)) - self.STABILITY
            return theta
        except Exception as e:
            print(e)

    
    def fit(self, data: dict, lr = 1e-6, epochs = 5, n_negatives = 5, burn_in = 8, burn_in_lr = 1e-3):
        """
        Fits Poincaré embeddings to the given data.

        Args:
            data (dict): A dictionary representing the training data.
            lr (float): Learning rate for optimization.
            epochs (int): Number of training epochs.
            n_negatives (int): Number of negative examples to generate.
            burn_in (int): Burn-in period for learning rate adjustment.
            burn_in_lr (float): Learning rate during the burn-in period.
        """

        for a in data:
            for b in data[a]:
                self.theta[b] = np.random.uniform(low=-0.001, high=0.001, size=(self.dim,))
            self.theta[a] = np.random.uniform(low=-0.001, high=0.001, size=(self.dim,))
        self.vocab = list(self.theta.keys())

        for a in self.theta:
            if not a in data:
                data[a] = []

        # Start training
        for epoch in range(epochs):
            print(f"Epoca: {epoch}")
            if epoch < burn_in:
                lr = burn_in_lr
            distances = []
            loss_negs = []
            for pos1 in self.vocab:
                if not data[pos1]:
                    continue
                pos2 = random.choice(data[pos1])
                negatives = self.generate_negative_examples(data, pos1, n_negatives)
                loss_den = self.calculate_denominator(negatives[1])
                loss_negs.append(loss_den) 
                distances.append(self.distance(self.theta[pos1], self.theta[pos2]))
                gradient = self.gradient(negatives=negatives[0], negative_distances=negatives[1], pos1=pos1, pos2=pos2, loss_denominator=loss_den)
                self.theta[pos1] = self.update_embedding(self.theta[pos1], -gradient[0], lr)
                self.theta[pos2] = self.update_embedding(self.theta[pos2], -gradient[1], lr)
                for (neg, der_neg) in zip(negatives[0], gradient[2]):
                    self.theta[neg[0]] = self.update_embedding(self.theta[neg[0]], -1*der_neg[0], lr)
                    self.theta[neg[1]] = self.update_embedding(self.theta[neg[1]], -1*der_neg[1], lr)

            real_loss = np.sum(distances - np.log(loss_negs))
            print(f"\tLoss: {real_loss}")