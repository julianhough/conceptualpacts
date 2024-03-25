#!pip install statsmodels
from statsmodels.stats.contingency_tables import mcnemar

def get_contingency_table(cl1_preds, cl2_preds, ground_truth):
    # returns a table of how many instances classifiers cl1 and cl2 both or each get (in)correct
    # Y = correct, N = incorrect
    # returns [[cl1Y+cl2Y, cl1Y+cl2N], [cl1N+cl2Y, cl1N+cl2N]]
    table = [[0,0], [0,0]]
    for c1, c2, y in zip(cl1_preds, cl2_preds, ground_truth):
        if c1 == y:
            if c2 == y:
                table[0][0]+=1
            else:
                table[0][1]+=1
        else:
            if c2 == y:
                table[1][0]+=1
            else:
                table[1][1]+=1
    return table

def calculate_mcnemar_test(c11_preds, c12_preds, ground_truth, alpha=0.05, exact=True, target_class=None):
    """Returns statistic, p-value, and whether signifcant according to alpha.
    Cite: McNemar, Quinn (June 18, 1947). "Note on the sampling error of the difference between correlated proportions or percentages". Psychometrika. 12 (2): 153–157
    And its application to ML:
    Dietterich, Thomas G. "Approximate statistical tests for comparing supervised classification learning algorithms." Neural computation 10, no. 7 (1998): 1895-1923.
    """
    if not target_class is None:
        # convert to one-vs-rest for this target class
        new_cl1_preds = []
        new_cl2_preds = []
        new_ground_truth = []
        for cl1_pred, cl2_pred, gt in zip(c11_preds, c12_preds, ground_truth):
            new_cl1_preds.append(1 if cl1_pred == target_class else 0)
            new_cl2_preds.append(1 if cl2_pred == target_class else 0)
            new_ground_truth.append(1 if gt == target_class else 0)
        c11_preds, c12_preds, ground_truth = new_cl1_preds, new_cl2_preds, new_ground_truth
            
    table = get_contingency_table(c11_preds, c12_preds, ground_truth)
    result = mcnemar(table, exact=exact)
    # summarize the finding
    print('statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))
    # interpret the p-value
    if result.pvalue > alpha:
        print('Same proportions of errors (fail to reject H0)')
    else:
        print('Different proportions of errors (reject H0)')
    return result.statistic, result.pvalue, result.pvalue <= alpha, result.pvalue <= 0.01, result.pvalue <= 0.001