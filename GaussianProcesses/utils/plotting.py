import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class MultivariateGaussian:

    def __init__(self, mean_vec=0, cov_matrix=0, x_range=0):
        self.mean_vec = mean_vec
        self.cov_matrix = cov_matrix
        self.x_range = x_range

    def plot_functions(self, n_functions):
        # Draw random vectors from this multivariate gaussian
        for ii in range(n_functions):

            y1 = np.random.multivariate_normal(self.mean_vec, self.cov_matrix)

            # Plot these randomly drawn functions
            plt.plot(self.x_range, y1)

        # Also show the true mean and covariance of prior
        std_vec = np.sqrt(np.diag(self.cov_matrix))
        plt.plot(self.x_range, self.mean_vec, c='green')
        plt.fill_between(self.x_range
                         , self.mean_vec - 1.96 * std_vec
                         , self.mean_vec + 1.96 * std_vec
                         , alpha=0.4
                         , edgecolor='green')
        # sns.lineplot(x = 'x_range', y = 'y_range', data= prior_df)
        plt.show()

    def plot_functions_mean(self, n_functions):
        # Draw random vectors from this multivariate gaussian
        prior_df = pd.DataFrame(columns=['x_range', 'y_range'])

        for ii in range(n_functions):

            y1 = np.random.multivariate_normal(self.mean_vec, self.cov_matrix)

            add_df = pd.DataFrame()
            add_df['x_range'] = self.x_range
            add_df['y_range'] = y1

            prior_df = prior_df.append(add_df)
        # Show the distribution and mean value for these functions

        sns.lineplot(x='x_range', y='y_range', data=prior_df)
        plt.show()