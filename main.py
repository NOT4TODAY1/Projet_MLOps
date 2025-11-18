import argparse
import os
from model_pipeline import train_all_models, prepare_data, load_model


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', type=str, default='alzheimers_disease_data.csv')
    p.add_argument('--train', action='store_true')
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--runall', action='store_true')
    p.add_argument('--models-dir', type=str, default='models')
    p.add_argument('--results-dir', type=str, default='results')
    p.add_argument('--load-model', type=str)
    return p


def main():
    p = build_parser()
    args = p.parse_args()

    if args.runall:
        print('Running full pipeline...')
        print('Step 1: Preparing data...')
        X_train, X_test, y_train, y_test = prepare_data(args.csv)
        print('X_train', getattr(X_train, 'shape', 'N/A'))
        print('X_test', getattr(X_test, 'shape', 'N/A'))
        print('y_train', getattr(y_train, 'shape', 'N/A'))
        print('y_test', getattr(y_test, 'shape', 'N/A'))
        print('Step 2: Training all models...')
        results_df = train_all_models(csv_path=args.csv, results_dir=args.results_dir, models_dir=args.models_dir)
        print('\nFinal Results:')
        print(results_df)
        print(f'\nAll files saved to models/ and results/')
        return

    if args.prepare:
        X_train, X_test, y_train, y_test = prepare_data(args.csv)
        print('X_train', getattr(X_train, 'shape', 'N/A'))
        print('X_test', getattr(X_test, 'shape', 'N/A'))
        print('y_train', getattr(y_train, 'shape', 'N/A'))
        print('y_test', getattr(y_test, 'shape', 'N/A'))
        return

    if args.train:
        print('Training...')
        results_df = train_all_models(csv_path=args.csv, results_dir=args.results_dir, models_dir=args.models_dir)
        print(results_df)
        return

    if args.load_model:
        m = load_model(args.load_model)
        print('Loaded', m)
        return

    p.print_help()


if __name__ == '__main__':
    main()
