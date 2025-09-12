''' # Normality: Q-Q Plot and Shapiro-Wilk
    qq = stats.probplot(model.resid, dist="norm")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=qq[0][0], y=qq[0][1], mode='markers', marker=dict(color='blue', size=8)))
    fig2.add_trace(go.Scatter(x=qq[0][0], y=qq[0][0], mode='lines', line=dict(color='red', dash='dash')))
    fig2.update_layout(title='Normality of Errors: Q-Q Plot',
                    xaxis_title='Theoretical Quantiles',
                    yaxis_title='Sample Quantiles')
   # fig2.show()

       # Linearity: Residuals vs Fitted
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=model.fittedvalues, y=model.resid, mode='markers', marker=dict(color='blue', size=8)))
    fig.add_hline(y=0, line_dash='dash', line_color='red')
    fig.update_layout(title='Residuals vs Adjusted Values', xaxis_title='Adjusted Values',
                    yaxis_title='Residuals',
                    yaxis_range=[min(model.resid)*1.1, max(model.resid)*1.1])
    #fig.show()

        # Forest Plot
    graph = fig_forest_plot(
        df=summary_df,
        labels=summary_df.columns.tolist(),
        only_display=True
    )

     graph = fig_forest_plot(
        df = logistic_response,
        labels = logistic_response.columns.tolist(),
        only_display=True
    )

        from sklearn.metrics import roc_auc_score, roc_curve
    import matplotlib.pyplot as plt



    y_scores = model.predict()
    auc = roc_auc_score(y, y_scores)
    print(f"ROC AUC Score: {auc:.3f}")


    fpr, tpr, thresholds = roc_curve(y, y_scores)
    plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()
'''