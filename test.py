import numpy as np

def compute_challenge_score(labels, outputs):
    assert len(labels) == len(outputs)
    num_instances = len(labels)

    # Use the unique output values as the thresholds for the positive and negative classes.
    thresholds = np.unique(outputs)
    thresholds = np.append(thresholds, thresholds[-1]+1)
    thresholds = thresholds[::-1]
    num_thresholds = len(thresholds)

    idx = np.argsort(outputs)[::-1]
    print(idx)
    # Initialize the TPs, FPs, FNs, and TNs with no positive outputs.
    tp = np.zeros(num_thresholds)
    fp = np.zeros(num_thresholds)
    fn = np.zeros(num_thresholds)
    tn = np.zeros(num_thresholds)

    tp[0] = 0
    fp[0] = 0
    fn[0] = np.sum(labels == 1)
    tn[0] = np.sum(labels == 0)
    # print(labels)
    # print(fn[0])
    # print(tn[0])

    # Update the TPs, FPs, FNs, and TNs using the values at the previous threshold.
    i = 0
    for j in range(1, num_thresholds):
        tp[j] = tp[j-1]
        fp[j] = fp[j-1]
        fn[j] = fn[j-1]
        tn[j] = tn[j-1]

        while i < num_instances and outputs[idx[i]] >= thresholds[j]:
            if labels[idx[i]]:
                tp[j] += 1
                fn[j] -= 1
            else:
                fp[j] += 1
                tn[j] -= 1
            i += 1

    # Compute the TPRs and FPRs.
    tpr = np.zeros(num_thresholds)
    fpr = np.zeros(num_thresholds)
    for j in range(num_thresholds):
        if tp[j] + fn[j] > 0:
            tpr[j] = float(tp[j]) / float(tp[j] + fn[j])
            fpr[j] = float(fp[j]) / float(fp[j] + tn[j])
        else:
            tpr[j] = float('nan')
            fpr[j] = float('nan')

    # Find the largest TPR such that FPR <= 0.05.
    max_fpr = 0.05
    max_tpr = float('nan')
    # print (fpr[0])
    if np.any(fpr <= max_fpr):
        indices = np.where(fpr <= max_fpr)
        max_tpr = np.max(tpr[indices])

    return max_tpr

y_true = np.array([1, 0, 1, 1, 0, 1, 0])
y_pred = np.array([1, 0, 1, 1, 0, 0, 0])
y_proba_pred = np.array([0.518, 0.493, 0.596, 0.521, 0.46, 0.49, 0.068])
challenge_score = compute_challenge_score(y_true, y_proba_pred)
print('Challenge Score: {:.3f}\n'.format(challenge_score))

# ROC Curve: The ROC curve is a graphical representation that shows the trade-off between sensitivity and 
# specificity across the entire range of threshold values. It provides a broader view of the model's performance without imposing any specific constraint on the FPR.

# Challenge Score: The challenge score is a single scalar metric that identifies the optimal threshold that 
# maximizes TPR while keeping the FPR at or below a predefined value (0.05). It provides a specific evaluation under a fixed false positive rate constraint.