from sklearn.ensemble import IsolationForest


class Utility:
    def __init__(self):
        pass

    def filter_outliers_iforest(
        self, df, contamination=0.1, random_state=42, n_jobs=-1
    ):
        iforest = IsolationForest(
            contamination=contamination, random_state=random_state, n_jobs=n_jobs
        )
        is_inlier = iforest.fit_predict(df[["y"]])
        filtered_df = df[is_inlier == 1]
        return filtered_df
