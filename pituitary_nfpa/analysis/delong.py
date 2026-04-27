import numpy as np
from scipy.stats import norm
from typing import Literal


def delong_test(
    labels: np.ndarray,
    pred1: np.ndarray,
    pred2: np.ndarray,
    alternative: Literal["two-sided", "greater", "lower"] = "two-sided",
    eps: float = 1e-8,
    conf_level: float = 0.95,
    ci_mode: Literal[
        "gaussian", "random-bootstrap", "stratified-bootstrap"
    ] = "gaussian",
) -> dict[str, float | np.ndarray]:
    """Implementation of DeLong statistical test for AUROCs."""
    X1, Y1 = pred1[labels == 1], pred1[labels == 0]
    X2, Y2 = pred2[labels == 1], pred2[labels == 0]

    # aucs
    theta_hat1 = ((X1[:, None] > Y1) + 0.5 * (X1[:, None] == Y1)).mean()
    theta_hat2 = ((X2[:, None] > Y2) + 0.5 * (X2[:, None] == Y2)).mean()

    if ci_mode == "gaussian":
        # V matrix
        V_X1 = ((X1[:, None] > Y1).astype(float) + 0.5 * (X1[:, None] == Y1)).mean(
            axis=1
        )
        V_Y1 = ((X1[:, None] > Y1).astype(float) + 0.5 * (X1[:, None] == Y1)).mean(
            axis=0
        )

        V_X2 = ((X2[:, None] > Y2).astype(float) + 0.5 * (X2[:, None] == Y2)).mean(
            axis=1
        )
        V_Y2 = ((X2[:, None] > Y2).astype(float) + 0.5 * (X2[:, None] == Y2)).mean(
            axis=0
        )

        # variances
        V_theta1 = np.var(V_X1, ddof=1) / len(X1) + np.var(V_Y1, ddof=1) / len(Y1)
        V_theta2 = np.var(V_X2, ddof=1) / len(X2) + np.var(V_Y2, ddof=1) / len(Y2)

        # covariances
        C = np.cov(V_X1, V_X2, ddof=1)[0, 1] / len(X1) + np.cov(V_Y1, V_Y2, ddof=1)[
            0, 1
        ] / len(Y1)

        ci_z = norm.ppf(1 - (1 - conf_level) / 2)

        # AUC 1
        se1 = np.sqrt(V_theta1)
        auc1 = np.array([theta_hat1 - ci_z * se1, theta_hat1, theta_hat1 + ci_z * se1])

        # AUC 2
        se2 = np.sqrt(V_theta2)
        auc2 = np.array([theta_hat2 - ci_z * se2, theta_hat2, theta_hat2 + ci_z * se2])

    else:

        n = 1000

        bootstrap1: list[float] = []
        i = 0
        while len(bootstrap1) < n:
            if ci_mode == "random-bootstrap":
                idx = np.random.choice(
                    np.arange(len(labels)), size=len(labels), replace=True
                )

            elif ci_mode == "stratified-bootstrap":
                pos_idx = np.where(labels == 1)[0]
                neg_idx = np.where(labels == 0)[0]

                pos_idx = np.random.choice(pos_idx, size=len(pos_idx), replace=True)
                neg_idx = np.random.choice(neg_idx, size=len(neg_idx), replace=True)

                idx = np.concatenate([pos_idx, neg_idx])

            bootstrap_labels = labels[idx]
            bootstrap_pred = pred1[idx]

            X, Y = (
                bootstrap_pred[bootstrap_labels == 1],
                bootstrap_pred[bootstrap_labels == 0],
            )

            if len(X) >= 1 and len(Y) >= 1:
                bootstrap1.append(((X[:, None] > Y) + 0.5 * (X[:, None] == Y)).mean())

            # prevent infinite loop after enough failed samples
            else:
                i += 1

            if i > 100 * n:
                print(f"Broke after {i} failed samples")
                break

        auc1 = np.array(
            [
                np.clip(
                    np.percentile(np.sort(bootstrap1), 100 * ((1 - conf_level) / 2)),
                    0,
                    1,
                ),
                np.clip(theta_hat1, 0, 1),
                np.clip(
                    np.percentile(
                        np.sort(bootstrap1), 100 * (1 - ((1 - conf_level) / 2))
                    ),
                    0,
                    1,
                ),
            ]
        )

        bootstrap2: list[float] = []
        i = 0
        while len(bootstrap2) < n:
            if ci_mode == "random-bootstrap":
                idx = np.random.choice(
                    np.arange(len(labels)), size=len(labels), replace=True
                )

            elif ci_mode == "stratified-bootstrap":
                pos_idx = np.where(labels == 1)[0]
                neg_idx = np.where(labels == 0)[0]

                pos_idx = np.random.choice(pos_idx, size=len(pos_idx), replace=True)
                neg_idx = np.random.choice(neg_idx, size=len(neg_idx), replace=True)

                idx = np.concatenate([pos_idx, neg_idx])

            bootstrap_labels = labels[idx]
            bootstrap_pred = pred2[idx]

            X, Y = (
                bootstrap_pred[bootstrap_labels == 1],
                bootstrap_pred[bootstrap_labels == 0],
            )

            if len(X) >= 1 and len(Y) >= 1:
                bootstrap2.append(((X[:, None] > Y) + 0.5 * (X[:, None] == Y)).mean())

            # prevent infinite loop after enough failed samples
            else:
                i += 1

            if i > 100 * n:
                print(f"Broke after {i} failed samples")
                break

        auc2 = np.array(
            [
                np.clip(
                    np.percentile(np.sort(bootstrap2), 100 * ((1 - conf_level) / 2)),
                    0,
                    1,
                ),
                np.clip(theta_hat2, 0, 1),
                np.clip(
                    np.percentile(
                        np.sort(bootstrap2), 100 * (1 - ((1 - conf_level) / 2))
                    ),
                    0,
                    1,
                ),
            ]
        )

        V_theta1 = np.var(bootstrap1, ddof=1)
        V_theta2 = np.var(bootstrap2, ddof=1)

        C = np.cov(bootstrap1, bootstrap2)[0, 1]

    # Compute the DeLong test statistic
    z = (theta_hat1 - theta_hat2) / np.sqrt(eps + V_theta1 + V_theta2 - 2 * C)

    if alternative == "two-sided":
        p_value = 2 * (1 - norm.cdf(abs(z)))

    elif alternative == "greater":
        p_value = 1 - norm.cdf(z)

    elif alternative == "lower":
        p_value = norm.cdf(z)

    else:
        raise ValueError(
            f"Alternative {alternative} is not supported. "
            "Select from ['two-sided', 'greater', 'lower']."
        )

    return {"p": p_value.item(), "auc1": auc1, "auc2": auc2}
