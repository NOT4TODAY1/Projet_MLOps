from src.data import load_data, split_features_target
from src.preprocess import clean_dataframe, scale_features
from src.models import get_models_and_grids
from src.train import train_and_evaluate


def main():
    csv_path = 'alzheimers_disease_data.csv'

    # Load
    df = load_data(csv_path)

    # Clean + preprocess
    df = clean_dataframe(df)
    df = scale_features(df)

    # Split
    X, y = split_features_target(df, target_col='Diagnosis')
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40, shuffle=True)

    # Models
    models, param_grids = get_models_and_grids()

    # Train & evaluate
    results_df = train_and_evaluate(models, param_grids, X_train, y_train, X_test, y_test)
    print(results_df)


if __name__ == '__main__':
    main()
