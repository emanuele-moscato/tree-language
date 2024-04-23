import os
import matplotlib.pyplot as plt
import seaborn as sns


sns.set_theme()


def plot_training_history(
        training_history,
        baseline_accuracy=None,
        savefig_dir=None,
        exp_id=None
    ):
    """
    Plots the training and validation losses along the training history
    (epochs).
    """
    # Plot loss.
    fig = plt.figure(figsize=(14, 6))

    sns.lineplot(
        x=range(len(training_history['training_loss'])),
        y=training_history['training_loss'],
        label='Training loss'
    )

    plot_title = 'Loss VS epoch'

    if 'val_loss' in training_history.keys():
        sns.lineplot(
            x=range(len(training_history['val_loss'])),
            y=training_history['val_loss'],
            label='Validation loss'
        )

        plot_title += f'\nFinal val loss: {training_history["val_loss"][-1]}'

    plt.title(plot_title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')

    if (savefig_dir is not None) and (exp_id is not None):
        plt.savefig(
            os.path.join(savefig_dir, f'{exp_id}_loss.png'),
            dpi=400,
            bbox_inches='tight'
        )

    # Plot accuracy.
    accuracy_metrics = [
        k for k in training_history.keys() if 'accuracy' in k or 'acc' in k
    ]

    if len(accuracy_metrics) > 0:
        fig = plt.figure(figsize=(14, 6))

        for i, accuracy_metric in enumerate(accuracy_metrics):
            sns.lineplot(
                x=range(len(training_history[accuracy_metric])),
                y=training_history[accuracy_metric],
                label=accuracy_metric,
                color=sns.color_palette()[i]
            )

        if baseline_accuracy is not None:
            if isinstance(baseline_accuracy, float):
                baseline_accuracy = [baseline_accuracy]

            if isinstance(baseline_accuracy, list):
                for j, ba in enumerate(baseline_accuracy):
                    sns.lineplot(
                        x=range(len(training_history[accuracy_metric])),
                        y=ba,
                        label='Baseline accuracy',
                        color=sns.color_palette()[i+j+1]
                    )
            elif isinstance(baseline_accuracy, dict):
                print('f')
                for j, (label, ba) in enumerate(baseline_accuracy.items()):
                    print(label, ba)
                    sns.lineplot(
                        x=range(len(training_history[accuracy_metric])),
                        y=ba,
                        label=label,
                        color=sns.color_palette()[i+j+1],
                    )

        plt.legend()

        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
    
        if (savefig_dir is not None) and (exp_id is not None):
            plt.savefig(
                os.path.join(savefig_dir, f'{exp_id}_accuracy.png'),
                dpi=400,
                bbox_inches='tight'
            )


def plot_evaluation(eval_results):
    """
    Given a results dataframe from the `evaluate_model` function, plots:
      * Predictions VS targets (compared with the ideal case).
      * Histogram for the distribution of residuals.
    """
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(14, 8))

    plt.subplots_adjust(hspace=0.4)

    sns.scatterplot(
        x=eval_results['target'],
        y=eval_results['pred'],
        color=sns.color_palette()[0],
        label='Actual predictions',
        ax=axs[0]
    )

    sns.lineplot(
        x=eval_results['target'],
        y=eval_results['target'],
        label='Ideal predictions',
        color=sns.color_palette()[1],
        ax=axs[0]
    )

    sns.lineplot(
        x=eval_results['target'],
        y=[eval_results['target'].mean()] * eval_results['target'].shape[0],
        label='Mean training target',
        color=sns.color_palette()[2],
        ax=axs[0]
    )

    plt.sca(axs[0])
    plt.legend(loc='upper left')
    plt.title('Predictions VS Target values')

    sns.histplot(
        x=eval_results['residual'],
        stat='density',
        ax=axs[1]
    )

    plt.sca(axs[1])
    plt.title('Distribution of residuals')
