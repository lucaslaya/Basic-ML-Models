import pandas as pd
import numpy as np

from Models.base_model import BaseModel

from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

class RandomForest(BaseModel):
    def __init__(self, train_df, test_df, categorical_features=[], random_seed=42):
        super().__init__(train_df, test_df, categorical_features, random_seed)

        # Model Specific parameter
        self.result_train_dict = {}
        self.result_val_dict = {}

        self.max_max_depth = 14
        self.max_n_estimators = 71

    def fit_and_validate(self):
        for m in range(1, self.max_max_depth, 2): # max depth of trees
            print("Current depth: ", m)
            for n in range(1, self.max_n_estimators, 2): # number of trees
                for s in [0.6, 0.7, 0.8, 0.9]:      # % of samples
                    for k in [0.6, 0.7, 0.8, 0.9]:  # % of features

                        tree_reg = RandomForestRegressor(random_state= 42, 
                                                        n_estimators= n, 
                                                        max_depth= m,
                                                        max_features= k,
                                                        bootstrap= True,
                                                        max_samples= s)

                        tree_reg.fit(self.X_train, np.ravel(self.y_train))

                        train_predicted = tree_reg.predict(self.X_train)
                        val_predicted = tree_reg.predict(self.X_val)

                        self.result_train_dict[m, n, s, k] = self.calculate_rmse(self.y_train, train_predicted)
                        self.result_val_dict[m, n, s, k] = self.calculate_rmse(self.y_val, val_predicted)

        return self.result_train_dict, self.result_val_dict

    def print_validation(self, top_n=10):
        results_list = [
            {'max_depth': k[0], 'n_estimators': k[1], 'max_samples': k[2], 'max_features': k[3], 'RMSE': v}
            for k, v in self.result_val_dict.items()
        ]
        results_df = pd.DataFrame(results_list)

        sorted_df = results_df.sort_values(by='RMSE', ascending=True).head(top_n)
        print(sorted_df)

    def plot_validation(self):
        fig, ax = plt.subplots(figsize=(15, 7))
        colors= ['black', 'red', 'blue', 'green', 'orange', 'gray','navy']
        color_index = 0
        custom_lines = np.array([])
        custom_names = np.array([])

        s=0.6
        k=0.8


        for m in range(1, self.max_max_depth, 2):

            line_plot_train = []
            line_plot_val = []

            for n in range(1, self.max_n_estimators, 2):
                line_plot_train.append(self.result_train_dict.get((m,n,s,k)))
                line_plot_val.append(self.result_val_dict.get((m,n,s,k)))

            # Un-comment this line for plotting train error evolution
            #plt.plot(np.arange(start=1, stop=self.max_n_estimators, step=2), 
            #         line_plot_train, alpha=0.5, c=colors[color_index], linestyle='--')

            # Un-comment this line for plotting validation error evolution
            plt.plot(np.arange(start=1, stop=self.max_n_estimators, step=2), 
                    line_plot_val, alpha=0.5, c=colors[color_index])

            color_line = Line2D([0], [0], color=colors[color_index], lw=4)
            custom_lines = np.append(custom_lines,color_line)
            color_name = 'max_depth =' + str(m)
            custom_names = np.append(custom_names,color_name)
            color_index+=1

        #ax.set_xlim(max_min_samples_leaf, 0)
        #plt.xticks(np.arange(start=1, stop=max_min_samples_leaf, step=2), rotation=90, size=10)

        ax.grid(True)
        plt.xlabel('# n_estimators')
        plt.ylabel('MSE-Log')

        ax.legend(custom_lines, custom_names)

        plt.show()

    def check_validation(self, n_estimators, max_depth, max_features, max_samples):
        tree_reg = RandomForestRegressor(
            random_state= self.random_seed,
            n_estimators= n_estimators,
            max_depth= max_depth,
            max_features= max_features,
            max_samples= max_samples,
            bootstrap= True
        )
        tree_reg.fit(self.X_train, np.ravel(self.y_train))

        train_predicted = tree_reg.predict(self.X_train)
        val_predicted = tree_reg.predict(self.X_val)

        print(f'y_train validated result: {self.calculate_rmse(self.y_train, train_predicted)}')
        print(f'y_val validated result: {self.calculate_rmse(self.y_val, val_predicted)}')

        tree_reg = RandomForestRegressor(
            random_state= self.random_seed,
            n_estimators= n_estimators,
            max_depth= max_depth,
            max_features= max_features,
            max_samples= max_samples,
            bootstrap= True
        )
        tree_reg.fit(self.X, np.ravel(self.y))

        train_predicted = tree_reg.predict(self.X)

        print(f'y validated result: {self.calculate_rmse(self.y, train_predicted)}')

        self.model = tree_reg
        return self.model
