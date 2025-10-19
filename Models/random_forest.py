import pandas as pd
import numpy as np

from Models.base_model import BaseModel

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

class RandomForest(BaseModel):
    def __init__(self, train_df, test_df, categorical_features=[], random_seed=42):
        super().__init__(train_df, test_df, categorical_features, random_seed)

        # Model Specific parameter
        self.result_train_dict = {}
        self.result_val_dict = {}
        self.grid_search = None
        self.cv_results_df = None

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
                                                        max_samples= s,
                                                        n_jobs=-1
                                                        )

                        tree_reg.fit(self.X_train, np.ravel(self.y_train))

                        train_predicted = tree_reg.predict(self.X_train)
                        val_predicted = tree_reg.predict(self.X_val)

                        self.result_train_dict[m, n, s, k] = self.calculate_rmse(self.y_train, train_predicted)
                        self.result_val_dict[m, n, s, k] = self.calculate_rmse(self.y_val, val_predicted)

        return self.result_train_dict, self.result_val_dict

    def fit_and_validate_optimized(self, n_jobs=-1, verbose=2):
        """
        Parallel grid search using GridSearchCV - MUCH FASTER than fit_and_validate().

        Args:
            n_jobs: Number of parallel jobs (-1 uses all cores)
            verbose: Verbosity level (0=silent, 1=minimal, 2=detailed, 3=very detailed)
        """
        # Define parameter grid
        param_grid = {
            'max_depth': list(range(1, self.max_max_depth, 2)),
            'n_estimators': list(range(1, self.max_n_estimators, 2)),
            'max_samples': [0.6, 0.7, 0.8, 0.9],
            'max_features': [0.6, 0.7, 0.8, 0.9],
            'bootstrap': [True]
        }

        # Create base estimator
        rf = RandomForestRegressor(random_state=self.random_seed)

        # Combine train and validation sets for GridSearchCV
        X_combined = pd.concat([self.X_train, self.X_val]).reset_index(drop=True)
        y_combined = np.concatenate([self.y_train, self.y_val])

        # Create custom CV split indices to match the original train/val split
        train_indices = np.arange(len(self.X_train))
        val_indices = np.arange(len(self.X_train), len(X_combined))
        cv_split = [(train_indices, val_indices)]

        # Create GridSearchCV
        # Note: We use negative MSE because GridSearchCV maximizes scores
        self.grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            scoring='neg_mean_squared_error',  # Will be converted to RMSE later
            cv=cv_split,  # Use our custom train/val split
            n_jobs=n_jobs,
            verbose=verbose,
            return_train_score=True
        )

        # Fit the grid search
        total_combinations = (len(param_grid['max_depth']) *
                            len(param_grid['n_estimators']) *
                            len(param_grid['max_samples']) *
                            len(param_grid['max_features']))
        print(f"Starting grid search with {total_combinations} parameter combinations...")
        print(f"Using {n_jobs if n_jobs > 0 else 'all available'} CPU cores\n")

        self.grid_search.fit(X_combined, y_combined)

        # Extract results and convert to your original format
        results = self.grid_search.cv_results_

        for i in range(len(results['params'])):
            params = results['params'][i]
            m = params['max_depth']
            n = params['n_estimators']
            s = params['max_samples']
            k = params['max_features']

            # Convert negative MSE to RMSE (assuming log-transformed target)
            # Note: GridSearchCV returns negative MSE, so we negate and sqrt
            val_rmse = np.sqrt(-results['mean_test_score'][i])
            train_rmse = np.sqrt(-results['mean_train_score'][i])

            self.result_train_dict[m, n, s, k] = train_rmse
            self.result_val_dict[m, n, s, k] = val_rmse

        # Store results as DataFrame for easy analysis
        self.cv_results_df = pd.DataFrame({
            'max_depth': [p['max_depth'] for p in results['params']],
            'n_estimators': [p['n_estimators'] for p in results['params']],
            'max_samples': [p['max_samples'] for p in results['params']],
            'max_features': [p['max_features'] for p in results['params']],
            'train_RMSE': [np.sqrt(-score) for score in results['mean_train_score']],
            'val_RMSE': [np.sqrt(-score) for score in results['mean_test_score']],
            'fit_time': results['mean_fit_time']
        })

        print(f"\nGrid search complete!")
        print(f"Best parameters: {self.grid_search.best_params_}")
        print(f"Best validation RMSE: {np.sqrt(-self.grid_search.best_score_):.6f}")

        return self.result_train_dict, self.result_val_dict

    def print_validation(self, top_n=10):
        if self.cv_results_df is not None:
            # Use the DataFrame created by GridSearchCV (includes fit times)
            sorted_df = self.cv_results_df.sort_values(by='val_RMSE', ascending=True).head(top_n)
            print("\nTop performing models:")
            print(sorted_df[['max_depth', 'n_estimators', 'max_samples', 'max_features', 'val_RMSE', 'fit_time']])
        else:
            # Fallback to original implementation
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
            bootstrap= True,
            n_jobs=-1
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
            bootstrap= True,
            n_jobs=-1
        )
        tree_reg.fit(self.X, np.ravel(self.y))

        train_predicted = tree_reg.predict(self.X)

        print(f'y validated result: {self.calculate_rmse(self.y, train_predicted)}')

        self.model = tree_reg
        return self.model
