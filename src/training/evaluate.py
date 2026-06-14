def evaluate():

    os.makedirs("results/plots",   exist_ok=True)
    os.makedirs("results/metrics", exist_ok=True)

    device = torch.device("cpu")
    print("=" * 55)
    print("MODEL EVALUATION — CNN BASELINE — TEST SET")
    print("=" * 55)

    _, _, test_loader = get_dataloaders(
        batch_size=32, num_workers=0
    )
    print(f"Test samples: {len(test_loader.dataset):,}")

    baseline_path = "models_saved/cnn_baseline_best.pth"
    if not os.path.exists(baseline_path):
        print("❌ Model not found!")
        return

    print("\nLoading CNN Baseline...")
    model = MedicalForgeryDetectorCNN()
    ckpt  = torch.load(baseline_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    preds, probs, labels = get_predictions(
        model, test_loader, device, dual_stream=False
    )

    results = print_metrics(labels, preds, probs, "CNN Baseline")

    plot_confusion_matrix(
        labels, preds,
        "CNN Baseline",
        "results/plots/confusion_baseline.png"
    )

    # ROC curve
    fig, ax = plt.subplots(figsize=(7, 6))
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc     = auc(fpr, tpr)
    ax.plot(fpr, tpr, color='blue', lw=2,
            label=f'CNN Baseline (AUC = {roc_auc:.3f})')
    ax.plot([0,1],[0,1],'k--', lw=1, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve — CNN Baseline')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/plots/roc_curve.png",
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  📊 ROC curve → results/plots/roc_curve.png")

    save_metrics_txt(
        [results],
        "results/metrics/model_evaluation.txt"
    )
    print("\n✅ Evaluation complete!")