import pandas as pd
import numpy as np

from Models.base_model import BaseModel

from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

class DecisionTree(BaseModel):
    def __init__(self, train_df, test_df, categorical_features=[], random_seed=42):
        super().__init__(train_df, test_df, categorical_features, random_seed)

        # Model specific parameter
        self.result_train_dict = {}
        self.result_val_dict = {}

        self.max_max_depth = 15
        self.max_min_samples_leaf = 50

    def fit_and_validate(self):
        for m in range(3, self.max_max_depth+1, 2):
            for n in range(1, self.max_min_samples_leaf+1, 3):
                tree_reg = DecisionTreeRegressor(
                    random_state=self.random_seed,
                    min_samples_leaf = n,
                    max_depth = m
                )

                tree_reg.fit(self.X_train, self.y_train)

                train_predicted = tree_reg.predict(self.X_train)
                val_predicted = tree_reg.predict(self.X_val)

                self.result_train_dict[m, n] = self.calculate_rmse(self.y_train, train_predicted)
                self.result_val_dict[m, n] = self.calculate_rmse(self.y_val, val_predicted)

        return self.result_train_dict, self.result_val_dict

    def print_validation(self, top_n=10):
        # Convert to DataFrame
        results_list = [
            {'max_depth': k[0], 'min_samples_leaf': k[1], 'RMSE': v}
            for k, v in self.result_val_dict.items()
        ]
        results_df = pd.DataFrame(results_list)

        # Sort by RMSE and get top N
        sorted_df = results_df.sort_values(by='RMSE', ascending=True).head(top_n)
        print(sorted_df)

    def plot_validation(self):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 7))
        colors= ['black', 'red', 'blue', 'green', 'orange', 'yellow', 'gray']
        color_index = 0
        custom_lines = np.array([])
        custom_names = np.array([])

        x_axis = np.arange(start=1, stop=self.max_min_samples_leaf+1, step=3)

        for m in range(3, self.max_max_depth+1, 2):

            line_plot_train = []
            line_plot_val = []

            for n in x_axis:
                line_plot_train.append(self.result_train_dict.get((m,n)))
                line_plot_val.append(self.result_val_dict.get((m,n)))

            ax1.plot(x_axis, line_plot_train, alpha=0.5, c=colors[color_index], linestyle='--')
            ax2.plot(x_axis, line_plot_val, alpha=0.5, c=colors[color_index])

            color_line = Line2D([0], [0], color=colors[color_index], lw=4)
            custom_lines = np.append(custom_lines,color_line)
            color_name = 'max_depth =' + str(m)
            custom_names = np.append(custom_names,color_name)
            color_index+=1

        ax1.set_xlim(self.max_min_samples_leaf, 0)
        ax2.set_xlim(self.max_min_samples_leaf, 0)

        plt.xticks(np.arange(start=1, stop=self.max_min_samples_leaf, step=2), rotation=90, size=10)

        ax1.grid(True)
        ax2.grid(True)

        plt.xlabel('# min_samples_leaf')
        plt.ylabel('RMSE')

        ax1.legend(custom_lines, custom_names)

        plt.show()

    def check_validation(self, min_samples_leaf, max_depth):
        tree_reg = DecisionTreeRegressor(
            random_state=self.random_seed, 
            min_samples_leaf=min_samples_leaf, 
            max_depth=max_depth
        )
        tree_reg.fit(self.X_train, self.y_train)

        train_predicted = tree_reg.predict(self.X_train)
        val_predicted = tree_reg.predict(self.X_val)

        print(f'y_train validated result: {self.calculate_rmse(self.y_train, train_predicted)}')
        print(f'y_val validated result: {self.calculate_rmse(self.y_val, val_predicted)}')

        tree_reg = DecisionTreeRegressor(
            random_state=self.random_seed,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth
        )
        tree_reg.fit(self.X, self.y)

        train_predicted = tree_reg.predict(self.X)

        print(f'y validated result: {self.calculate_rmse(self.y, train_predicted)}')

        self.model = tree_reg
        return self.model
