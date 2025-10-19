import matplotlib.pyplot as plt

def generate_roc_plot(fprs, tprs, roc_aucs):
    plt.figure(figsize=(10, 8))

    for model_name in fprs.keys():
        plt.plot(
            fprs[model_name],
            tprs[model_name],
            lw=2,
            label=f"{model_name} (AUC = {roc_aucs[model_name]:.2f})"
        )

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("ROC Curves (Macro-Average) - Multiclass Classification (glass_identification)")
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("roc_multiclass_glass.png")