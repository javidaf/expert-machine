import os
from ml_p2.regression.logistic_regression.logistic_regression import (
    LogisticRegression,
    run_logistic_regression_comparison,
)


if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)

    chdata_file = os.path.join(script_dir, "data", "chddata.csv")

    chdata_file = os.path.normpath(chdata_file)

    run_logistic_regression_comparison(chdata_file)
