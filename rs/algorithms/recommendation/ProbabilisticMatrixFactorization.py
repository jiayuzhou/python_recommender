
import numpy as np


class ProbabilisticMatrixFactorization():

    def __init__(self, rating_tuples, latent_d=1, lamb = 0.1, learning_rate = 0.0001):
        self.latent_d = latent_d;
        self.learning_rate = learning_rate;
        self.regularization_strength = lamb;
        
        self.ratings = np.array(rating_tuples).astype(float)
        self.converged = False

        self.num_users = int(np.max(self.ratings[:, 0]) + 1)
        self.num_items = int(np.max(self.ratings[:, 1]) + 1)
        
        print (self.num_users, self.num_items, self.latent_d)
        print self.ratings

        self.users = np.random.random((self.num_users, self.latent_d))
        self.items = np.random.random((self.num_items, self.latent_d))

        self.new_users = np.random.random((self.num_users, self.latent_d))
        self.new_items = np.random.random((self.num_items, self.latent_d))           


    def likelihood(self, users=None, items=None):
        if users is None:
            users = self.users
        if items is None:
            items = self.items
            
        sq_error = 0
        
        for rating_tuple in self.ratings:
            if len(rating_tuple) == 3:
                (i, j, rating) = rating_tuple
                weight = 1
            elif len(rating_tuple) == 4:
                (i, j, rating, weight) = rating_tuple
            
            r_hat = np.sum(users[int(i)] * items[int(j)])

            sq_error += weight * (rating - r_hat)**2

        L2_norm = 0
        for i in range(self.num_users):
            for d in range(self.latent_d):
                L2_norm += users[i, d]**2

        for i in range(self.num_items):
            for d in range(self.latent_d):
                L2_norm += items[i, d]**2

        return -sq_error - self.regularization_strength * L2_norm
        
        
    def update(self):

        updates_o = np.zeros((self.num_users, self.latent_d))
        updates_d = np.zeros((self.num_items, self.latent_d))        

        for rating_tuple in self.ratings:
            if len(rating_tuple) == 3:
                (i, j, rating) = rating_tuple
                weight = 1
            elif len(rating_tuple) == 4:
                (i, j, rating, weight) = rating_tuple
            
            r_hat = np.sum(self.users[i] * self.items[j])
            
            for d in range(self.latent_d):
                updates_o[i, d] += self.items[j, d] * (rating - r_hat) * weight
                updates_d[j, d] += self.users[i, d] * (rating - r_hat) * weight

        while (not self.converged):
            initial_lik = self.likelihood()

            print "  setting learning rate =", self.learning_rate
            self.try_updates(updates_o, updates_d)

            final_lik = self.likelihood(self.new_users, self.new_items)

            if final_lik > initial_lik:
                self.apply_updates(updates_o, updates_d)
                self.learning_rate *= 1.25

                if final_lik - initial_lik < .1:
                    self.converged = True
                    
                break
            else:
                self.learning_rate *= .5
                self.undo_updates()

            if self.learning_rate < 1e-10:
                self.converged = True

        return not self.converged
    

    def apply_updates(self, updates_o, updates_d):
        for i in range(self.num_users):
            for d in range(self.latent_d):
                self.users[i, d] = self.new_users[i, d]

        for i in range(self.num_items):
            for d in range(self.latent_d):
                self.items[i, d] = self.new_items[i, d]                

    
    def try_updates(self, updates_o, updates_d):        
        alpha = self.learning_rate
        beta = -self.regularization_strength

        for i in range(self.num_users):
            for d in range(self.latent_d):
                self.new_users[i,d] = self.users[i, d] + \
                                       alpha * (beta * self.users[i, d] + updates_o[i, d])
        for i in range(self.num_items):
            for d in range(self.latent_d):
                self.new_items[i, d] = self.items[i, d] + \
                                       alpha * (beta * self.items[i, d] + updates_d[i, d])
        

    def undo_updates(self):
        # Don't need to do anything here
        pass


    def print_latent_vectors(self):
        print "Users"
        for i in range(self.num_users):
            print i,
            for d in range(self.latent_d):
                print self.users[i, d],
            print
            
        print "Items"
        for i in range(self.num_items):
            print i,
            for d in range(self.latent_d):
                print self.items[i, d],
            print    


def fake_ratings(noise=.25):
    u = []
    v = []
    ratings = []
    
    num_users = 100
    num_items = 100
    num_ratings = 30
    latent_dimension = 10
    
    # Generate the latent user and item vectors
    for i in range(num_users):
        u.append(2 * np.random.randn(latent_dimension))
    for i in range(num_items):
        v.append(2 * np.random.randn(latent_dimension))
        
    # Get num_ratings ratings per user.
    for i in range(num_users):
        items_rated = np.random.permutation(num_items)[:num_ratings]

        for jj in range(num_ratings):
            j = items_rated[jj]
            rating = np.sum(u[i] * v[j]) + noise * np.random.randn()
        
            ratings.append((i, j, rating))  # thanks sunquiang

    return (ratings, u, v)





if __name__ == "__main__":

    DATASET = 'fake'

    if DATASET == 'fake':
        (ratings, true_o, true_d) = fake_ratings()

    #plot_ratings(ratings)

    pmf = ProbabilisticMatrixFactorization(ratings, latent_d=5)
    
    liks = []
    while (pmf.update()):
        lik = pmf.likelihood()
        liks.append(lik)
        print "L=", lik
        pass


    pmf.print_latent_vectors()
