import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

def reliability_curve(y_true, y_score, bins=10, normalize=True):
    """Compute reliability curve

    Reliability curves allow checking if the predicted probabilities of a
    binary classifier are well calibrated. This function returns two arrays
    which encode a mapping from predicted probability to empirical probability.
    For this, the predicted probabilities are partitioned into equally sized
    bins and the mean predicted probability and the mean empirical probabilties
    in the bins are computed. For perfectly calibrated predictions, both
    quantities should be approximately equal (for sufficiently many test
    samples).

    Note: this implementation is restricted to binary classification.
    Modified to return by APBradley (2017).

    Parameters
    ----------

    y_true : array, shape = [n_samples]
        True binary labels (0 or 1).

    y_score : array, shape = [n_samples]
        Target scores, can either be probability estimates of the positive
        class or confidence values. If normalize is False, y_score must be in
        the interval [0, 1]

    bins : int, optional, default=10
        The number of bins into which the y_scores are partitioned.
        Note: n_samples should be considerably larger than bins such that
              there is sufficient data in each bin to get a reliable estimate
              of the reliability

    normalize : bool, optional, default=True
        Whether y_score needs to be normalized into the bin [0, 1]. If True,
        the smallest value in y_score is linearly mapped onto 0 and the 
        largest one onto 1.

    Returns
    -------
    y_score_bin_mean : array, shape = [bins]
        The mean predicted y_score in the respective bins.

    empirical_prob_pos : array, shape = [bins]
        The empirical probability (frequency) of the positive class (+1) in the
        respective bins.

    References
    ----------
    .. [1] `Predicting Good Probabilities with Supervised Learning
            <http://machinelearning.wustl.edu/mlpapers/paper_files/icml2005_Niculescu-MizilC05.pdf>`_

    """
    # Normalize scores into bin [0, 1]
    if normalize:  
        y_score = (y_score - y_score.min()) / (y_score.max() - y_score.min())

    bin_width = 1.0 / bins
    bin_centers = np.linspace(0, 1.0 - bin_width, bins) + bin_width / 2

    y_score_bin_mean = np.zeros(bins)
    empirical_prob_pos = np.zeros(bins)
    for i, threshold in enumerate(bin_centers):
        # determine all samples where y_score falls into the i-th bin
        bin_idx = np.logical_and(threshold - bin_width / 2 < y_score,
                                 y_score <= threshold + bin_width / 2)
        # Store mean y_score and mean empirical probability of positive class
        if np.count_nonzero(bin_idx):
            y_score_bin_mean[i] = y_score[bin_idx].mean()
            empirical_prob_pos[i] = y_true[bin_idx].mean()
    
    return y_score_bin_mean, empirical_prob_pos

def pav(target, score):
    """
    PAV uses the pair adjacent violators algorithm to produce a monotonic
    (piecewise constant) smoothing of classifier scores. 
    This calibrates the scores to be calibrated posterior probabilities 
    and produces scores that form a ROC convex hull.
    
    Translated from matlab by Sean Collins (2006) as part of the EMAP toolbox
    Modified to sort based on score and target arrays (as per scikit) by APBradley (2017).
    
    For details see: 
    PAV and the ROC convex hull, Tom Fawcett and Alexandru Niculescu-Mizil, 
    Mach Learn (2007) 68: 97–106.
    
    Parameters
    ----------
    target : array, shape = [n_samples]
        True binary labels in range {0, 1} or {-1, 1}.

    score : array, shape = [n_samples]
        Target scores, can either be posterior probability estimates of the
        positive class, confidence values, or non-thresholded measure of
        decisions (as returned by a “decision_function” on some classifiers).
    
    Returns:
    t: target labels sorted according to the input scores
    v: sorted scores as calibrated probabilities (0,1)
    
    """
    s_ind = np.argsort(score)
    t = target[s_ind]
    y = target[s_ind]
    
    y = np.asarray(y)
    assert y.ndim == 1
    n_samples = len(y)
    v = y.copy()
    lvls = np.arange(n_samples)
    lvlsets = np.c_[lvls, lvls]
    flag = 1
    while flag:
        deriv = np.diff(v)
        if np.all(deriv >= 0):
            break

        viol = np.where(deriv < 0)[0]
        start = lvlsets[viol[0], 0]
        last = lvlsets[viol[0] + 1, 1]
        s = 0
        n = last - start + 1
        for i in range(start, last + 1):
            s += v[i]

        val = s / n
        for i in range(start, last + 1):
            v[i] = val
            lvlsets[i, 0] = start
            lvlsets[i, 1] = last
    
    return (t, v)

def plot_bland_altman(data1, data2, *args, **kwargs):
    """
    Function to draw a Bland Altman plot comparing two clinical measurements

    See Bland and Altman
    STATISTICAL METHODS FOR ASSESSING AGREEMENT BETWEEN TWO METHODS OF CLINICAL
    MEASUREMENT

    Example: plot_bland_altman(np.random(10), np.random(10))
    """
    data1     = np.asarray(data1)
    data2     = np.asarray(data2)
    mean      = np.mean([data1, data2], axis=0)
    diff      = data1 - data2                   # Difference between data1 and data2
    md        = np.mean(diff)                   # Mean of the difference
    sd        = np.std(diff, axis=0)            # Standard deviation of the difference

    plt.scatter(mean, diff, *args, **kwargs)
    plt.axhline(md,           color='gray', linestyle='--')
    plt.axhline(md + 1.96*sd, color='red', linestyle='--')
    plt.axhline(md - 1.96*sd, color='red', linestyle='--')
    plt.title('Bland-Altman Plot')
    plt.xlabel('Mean')
    plt.ylabel('Difference')
    plt.show()


def roc_curve(target, score, pos_label=None, sample_weight=None, drop_intermediate=False):
    """
    Mirror of roc_curve in sklearn.metrics with drop_intermediate defaulted to False
    This increases the accuracy when extimating partial_auc, by including redundant operating points

    Parameters
    ----------
    y_true : array, shape = [n_samples]
        True binary labels in range {0, 1} or {-1, 1}. If labels are not binary,
        pos_label should be explicitly given.

    y_score : array, shape = [n_samples]
        Target scores, can either be posterior probability estimates of the
        positive class, confidence values, or non-thresholded measure of
        decisions (as returned by “decision_function” on some classifiers).

    pos_label : int or str, default=None
        Label considered as positive and others are considered negative.

    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.

    drop_intermediate : boolean, optional (default=False)
        Whether to drop some suboptimal thresholds which would not appear on a
        plotted ROC curve. This is useful in order to create lighter ROC curves,
        i.e., with less operating points.

    Returns
    -------
    fpr : array, shape = [>2]
        Increasing false positive rates such that element i is the false positive
        rate of predictions with score >= thresholds[i].

    tpr : array, shape = [>2]
        Increasing true positive rates such that element i is the true positive
        rate of predictions with score >= thresholds[i].

    thresh : array, shape = [n_thresholds]
        Decreasing thresholds on the decision function (posterior) used to
        compute fpr and tpr. thresholds[0] represents no instances being
        predicted and is arbitrarily set to max(y_score) + 1.
    """
    fpr, tpr, thresh = metrics.roc_curve(target, score, pos_label,
                                         sample_weight, drop_intermediate)
    return (fpr, tpr, thresh)

def partial_auc(fpr, tpr, op1=0.0, op2=1.0, Sp=True):
    """
    Estimate the partial AUC between Se or Sp operating points op1 and op2
     Note: for pauc to be estimated accurately drop_intermediate=False in
     metrics.roc_curve

    Parameters
    ----------
    fpr : array, shape = [>2]
        Increasing false positive rates

    tpr : array, shape = [>2]
        Increasing true positive rates

    Op1,Op2 : float, optional (default = 0.0,1.0), i.e., whole curve full AUC
        Specificity or Sensitivity points between which to calculate partial
        auc (range 0,1)

    Sp : boolean, optional (default=True)
        Whether operating points specify a range on Specificty (TNR=1-FPR) or
        Sensitivity (TPR) i.e., a vertical (Sp) or horizontal (Se) partial AUC

    Return
    ------
    p_auc : float
        Partial AUC in between op1 and op2
    """

    if (op1 >= op2):
        raise ValueError('op1 must be less than op2')

    if (op1 > 0.0) or (op2 < 1.0):
        if Sp:
            # Constraints on Sp, vertical slice of ROC curve
            # find those operating points and zero out either side of them
            op1 = 1 - op1
            op2 = 1 - op2
            mask1 = np.greater_equal(fpr,np.ones(fpr.size)*op2)
            mask2 = np.less_equal(fpr,np.ones(fpr.size)*op1)
            fpr = fpr*np.logical_and(mask1,mask2)
            tpr = tpr*np.logical_and(mask1,mask2)
            p_auc = metrics.auc(fpr,tpr,reorder=True)
        else:
            # Constraints on Se, Calculate horizontal slice of ROC curve
            # By first find the Sp at op1 and op2 calculating veritcal area
            # adding in rectangular area to left of this and subtract area below
            i=0
            while tpr[i] < op1: i += 1

            Sp2 = 1 - fpr[i]

            i=len(tpr)-1
            while tpr[i] > op2: i -= 1

            Sp1 = 1 - fpr[i]
            p_auc = partial_auc(fpr,tpr,Sp1,Sp2,Sp=True)
            p_auc += ((op2-op1)*(Sp1)) - ((Sp2-Sp1)*op1)

    else:
        p_auc = metrics.auc(fpr,tpr)

    return p_auc

def neyman_pearson(fpr, tpr, thresh, min_rate=0.95, Se=True):
    """
    Function that finds the operating point (threshold posterior) on a ROC curve
    that maximises Sp given a constraint on on a minimum level of Sp (or vice versa)

    Parameters
    ----------
    fpr : array, shape = [>2]
        Increasing false positive rates

    tpr : array, shape = [>2]
        Increasing true positive rates

    thresh : array, shape = [n_thresholds]
        Decreasing thresholds on the decision function

    min_rate : float (0,1), optional (default=0.95)
        Constraint on sensitivity (tpr) or specificity (tnr) in range 0 to 1

    Se : boolean, optional (default=True)
        is min_rate a constraint on Se (True) or Sp (False)

    Returns
    -------
    np_fpr, np_tpr, np_thresh : float
        The operating point (fpr, tpr) that meets the constraint on min_rate and
        associated decision threshold
    """
    np_tpr = np_thresh = np_fpr = 0.0
    if Se:
        for i, tpr_val in enumerate(tpr):
            if tpr_val >= min_rate:
                np_tpr = tpr_val
                np_fpr = fpr[i]
                np_thresh = thresh[i]
                return (np_fpr, np_tpr, np_thresh)

    else:
        # enumerate a decreasing array of tnr
        for i, tnr_val in enumerate(1-fpr):
            if tnr_val < min_rate:
                np_fpr = fpr[i-1]
                np_tpr = tpr[i-1]
                np_thresh = thresh[i-1]
                return (np_fpr, np_tpr, np_thresh)

def chi_sqr_val(tpr, fpr, Nn, Np):
    """
    function to calculate chi sqaured value given:
    tpr, fpr : float
        The true positive and false positive rates (1-beta and alpha)
    Nn, Np : int
        The number of Negative and Positive samples in the test set

    Returns
    -------
    chi : float
        The chi squared value at this operating point

    """
    # convert from rates to values in the confusion matrix (contingency table)
    fn = (1-tpr)*Np
    tn = (1-fpr)*Nn
    tp = tpr*Np
    fp = fpr*Nn
    # count the number of positive and negative predicitions (marginals)
    rp = fp+tp
    rn = tn+fn
    # expected values
    etn = (rn*Nn)/(Nn+Np)+0.000001
    etp = (rp*Np)/(Nn+Np)+0.000001
    efn = (rn*Np)/(Nn+Np)+0.000001
    efp = (rp*Nn)/(Nn+Np)+0.000001
    # return the chi squared value
    chi = ((((tn-etn)**2)/etn)+(((tp-etp)**2)/etp)+(((fn-efn)**2)/efn) +(((fp-efp)**2)/efp))
    
    return chi

def max_npv(fpr, tpr, Nn, Np):
    """
    Finds the best Negative Predictive Value (NPV) and associated operating point
    NPV = TN/(TN+FN) - note depends of prevelance of negative class

    Parameters
    ----------
    fpr : array, shape = [n]
        False positive rates, i.e., x coordinates of ROC curve.
    tpr : array, shape = [n]
        True positive rates (sensitivity), i.e., y coordinates of ROC curve.
    Nn, Np : int
        The number of negative and positive samples in the dataset the ROC curve
        was constructed from

    Returns
    -------
    Bnpv, Bnpv_fpr, Bnpv_tpr: float
        The maximum NPV and the operating point (fpr, tpr) with max NPV
    """
    tnr = 1-fpr
    fnr = 1-tpr
    npv = np.zeros(len(tnr))
    Bnpv = 0.0
    for i, tnr_val in enumerate(tnr):
        if tnr_val == 0.0:
            npv[i] = 0.0
        else:
            npv[i] = (tnr_val*Nn)/((tnr_val*Nn) + (fnr[i]*Np))
            if npv[i] > Bnpv:
                Bnpv = npv[i]
                Bnpv_tpr = tpr[i]
                Bnpv_fpr = fpr[i]

    return (Bnpv, Bnpv_fpr, Bnpv_tpr)

def max_ppv(fpr, tpr, Nn, Np):
    """
    Finds the best Positive Predictive Value (PPV) and associated operating point
    PPV = TP/(TP+FP) - note depends of prevelance of positive class

    Parameters
    ----------
    fpr : array, shape = [n]
        False positive rates, i.e., x coordinates of ROC curve.
    tpr : array, shape = [n]
        True positive rates (sensitivity), i.e., y coordinates of ROC curve.
    Nn, Np : int
        The number of negative and positive samples in the dataset the ROC curve
        was constructed from

    Returns
    -------
    Bppv, Bppv_fpr, Bppv_tpr: float
        The maximum PPV and the operating point (fpr, tpr) with max PPV
    """

    ppv = np.zeros(len(tpr))
    Bppv = 0.0
    for i, tpr_val in enumerate(tpr):
        if tpr_val == 0.0:
            ppv[i] = 0.0
        else:
            ppv[i] = (tpr_val*Np)/((tpr_val*Np) + (fpr[i]*Nn))
            if ppv[i] >= Bppv:
                Bppv = ppv[i]
                Bppv_tpr = tpr[i]
                Bppv_fpr = fpr[i]

    return (Bppv, Bppv_fpr, Bppv_tpr)

def max_youden_J(fpr, tpr, thresh):
    """
    Finds the empirical maximum value of Youden's J statistic (TPR - FPR = Se + Sp - 1)
    and associated ROC point. Youden's J is the vertical distance from the by chance
    diagonal line to an operating point on the ROC curve.
    Also known as deltaP' and informedness in the multi-class case (i.e., > 2 classes)
    Note: Preferable to Cohen's Kappa when one of the raters is the gold standard (truth)
    see: Powers, David MW. "The problem with kappa." 13th Conference of the
    European Chapter of the Association for Computational Linguistics, 2012.

    Parameters
    ----------
    fpr : array, shape = [n]
        False positive rates, i.e., x coordinates of ROC curve.
    tpr : array, shape = [n]
        True positive rates (sensitivity), i.e., y coordinates of ROC curve.

    Returns
    -------
    Jval, Jfpr, Jtpr, Jthresh: float
        The maximum Youden's J and the associated operating point (fpr, tpr)
        and (posterior) decision threshold
    """

    Jval = Jtpr = Jtnr = Jthresh = 0.0
    tnr = 1-fpr
    # Traverse the ROC curve finding the point that maximise J (furthest away from diagonal)
    for i in range(len(tpr)):
        if (tpr[i] + tnr[i] - 1) > Jval:
            Jtpr = tpr[i]
            Jtnr = tnr[i]
            Jfpr = fpr[i]
            Jthresh = thresh[i]
            Jval = Jtnr + Jtpr - 1

    return (Jval, Jfpr, Jtpr, Jthresh)

def bayes_error(fpr, tpr, thresh, Nn, Np):
    """
    Finds the empirical Bayes error (minimum error rate) and associated ROC point

    Parameters
    ----------
    fpr : array, shape = [n]
        False positive rates, i.e., x coordinates of ROC curve.
    tpr : array, shape = [n]
        True positive rates (sensitivity), i.e., y coordinates of ROC curve.
    Nn, Np : int
        The number of negative and positive samples in the dataset the ROC curve
        was constructed from

    Returns
    -------
    Berror, Bfpr, Btpr, Bthresh: float
        The minimum error and the operating point (fpr, tpr) with minimum error
        and (posterior) decision threshold
    """

    BAcc = Btpr = Btnr = Bthresh = 0.0
    # convert rates into counts
    tpr = tpr*Np
    tnr = (1 - fpr)*Nn
    # Then traverse the ROC curve finding the point that makes the least errors
    for i in range(len(tpr)):
        if (tpr[i] + tnr[i]) > BAcc:
            Btpr = tpr[i]
            Btnr = tnr[i]
            Bthresh = thresh[i]
            BAcc = Btnr + Btpr

    # convert this to error and the rates (fpr, tpr) of that point
    Berror = 1 - (BAcc/(Nn+Np))
    Btpr = Btpr/Np
    Btnr = Btnr/Nn
    Bfpr = 1 - Btnr

    return (Berror, Bfpr, Btpr, Bthresh)

def sew_auc(AUC, nn, np):
    """
     function sew_auc(AUC, nn, np)
     Estimates the standard error of the area under the roc curve
     based on AUC and the number of positive/negative samples in the dataset.
     based on the standard error of the Wilcoxon test,
     See Hanley & McNeil 1982 "The meaning and use of AUC"

     Parameters
     ----------
     AUC = desired or achieved AUC (Wilcoxon P(p>n))
     nn = Number negative samples used to estimate AUC
     np = Number of positive samples used

     Return
     ------
     std_err : float
    """
    Q1 = AUC/(2-AUC)
    Q2 = (2*AUC**2)/(1+AUC)
    std_err = ((AUC*(1-AUC))+((np-1)*(Q1-AUC**2))+((nn-1)*(Q2-AUC**2)))/(nn*np)

    return std_err

def plot_roc(target, score, plot_type='SeSp', title=None, save_pdf=False, min_err=False,
             ppv_npv=False, n_p='', np_min=0.9, max_J=False,
             pos_label=None, sample_weight=None, drop_intermediate=True):
    """

    Plot and print a Receiver Operating Characteristic (ROC) curve
    Adds a title, a legend inculding AUC +/- standard error and saves a pdf

    Note this function is limited to binary classification tasks (dichotemisers)
    Uses sklearn.metrics.roc_curve and metrics.auc

    Parameters
    ----------
     target : array, shape = [n_samples]
         True binary labels in range {0, 1} or {-1, 1}.  If labels are not
         binary, pos_label should be explicitly given.

     score : array, shape = [n_samples]
         Target scores, can either be probability estimates of the positive
         class, confidence values, or non-thresholded measure of decisions
         (say, as returned by softmax).

     plot_type : str, optional (default='SeSp')
         The type of ROC to plot:
             'SeSp' Sensitivity (TPR) v Specificty (TNR = 1 - FPR)
             'ROC'  ROC curve true positive rate (TPR) v false positive rate (FPR)
             'PR'   Precision (PPV = TP/(TP+FP)) v Recall (TPR = Sensitivity)
             'IPR'  Inverse Precision-Recall,
                    i.e., Negative Predictive Value (NPV) v Specifity (TNR)
             'Chi'  ROC curve with Chi Squared contours where alpha = 0.05
                    critical value = 3.84

             NOTE : Both Precision (PPV) and its inverse (NPV) are class prior
                    (skew) dependent and so only make sense when the test set on
                    which they are measured has the "natural" priors expected in
                    population, i.e., not an "enriched" data set Chi is dependent
                    on both the number of positive and negative samples

     title : str, optional (default=None)
         Title to prepend to the figure title and pdf file (if saved)

     save_pdf : boolean, optional (default=False)
         Whether a pdf file of the figure is saved in current working directory

     min_err : boolean, optional (default=False)
         Whether to highlight the minimum error operating point

     ppv_npv : boolean, optional (default=False
         Whether to highlight the best NPV and PPV operating points

     n_p : str, optional (default=Empty)
         Whether to find and plot the Neyman-Pearson threshold that meets a
         minimum constraint on 'Se' or 'Sp'

     np_min : float, optional (default=0.95)
         Find the operating point that meets this minimum 'Se' or 'Sp' value

     max_J : boolean, optional (default=False)
         Whether to highlight the operating point with maximum Youden's J
         (AKA informedness or deltaP') i.e., (TPR - FPR) max vertical distance
         from the by-chance diagonal line

     pos_label : int or str, default=None
         Label considered as positive in target, others are considered negative.

     sample_weight : array-like of shape = [n_samples], optional
         Sample weights, default=None

     drop_intermediate : boolean, optional (default=True)
         Whether to drop some suboptimal thresholds which would not appear
         on a plotted ROC curve. This is useful in order to create lighter
         ROC curves.

    Returns
    --------
       fig, returns the figure handle and optionally saves it as a pdf

    Example
    --------

    plot_roc(np.array([0, 0, 0, 1, 1, 1]), np.array([0.0, 0.1, 0.4, 0.35, 0.8, 1.0]),
             plot_type='sesp', ppv_npv=True, min_err=True, n_p='Se', np_min=0.9)
    """

    #fpr, tpr, thresh = metrics.roc_curve(target, score, pos_label,
    #                                     sample_weight, drop_intermediate)
    # Don't drop intermediate operating points else partial AUC won't be
    # estimated accurately
    fpr, tpr, thresh = roc_curve(target, score, pos_label, sample_weight,
                                 drop_intermediate=False)
    roc_auc = partial_auc(fpr,tpr)
    # Total number of test samples
    N = len(target)
    # number of positive and negative samples
    Np = np.count_nonzero(target)
    Nn = N-Np
    sew = sew_auc(roc_auc, Nn, Np)
    th_np = 0.0
    if n_p.lower() == 'se':
        fpr_np, tpr_np, th_np = neyman_pearson(fpr,tpr,thresh,np_min,Se=True)
    elif n_p.lower() == 'sp':
        fpr_np, tpr_np, th_np = neyman_pearson(fpr,tpr,thresh,np_min,Se=False)

    # if you want to plot minimum error point figure out what and where it is
    if min_err:
        merr, mfpr, mtpr, mthresh = bayes_error(fpr,tpr,thresh,Nn,Np)

    if ppv_npv:
        Bppv, Bppv_fpr, Bppv_tpr = max_ppv(fpr, tpr, Nn, Np)
        Bnpv, Bnpv_fpr, Bnpv_tpr = max_npv(fpr, tpr, Nn, Np)

    # if you want to plot the maximum Youden's J point figure out where it is
    if max_J:
        Jval, Jfpr, Jtpr, Jthresh = max_youden_J(fpr,tpr,thresh)

    if title:
        title += ': Receiver Operating Characteristic'
        fname = title + '_ROC.pdf'
    else:
        title = 'Receiver Operating Characteristic'
        fname = 'ROC.pdf'

    # Ensure ROC curve goes all the way to (0,0)
    if tpr[0] != fpr[0]:
        tpr = np.insert(tpr,0,0.0)
        fpr = np.insert(fpr,0,0.0)

    # open a figure window and plot the curve
    fig, ax = plt.subplots()
    if plot_type.lower() == 'sesp':
        # Plot Sp V Se
        plt.plot(1-fpr, tpr,'b-', label='AUC = {:0.3f} +/-{:0.4f}'.format(roc_auc, sew))

        if ppv_npv:
            plt.plot(1-Bppv_fpr, Bppv_tpr,'ro', label='PPV = {:0.3f}'.format(Bppv))
            plt.plot(1-Bnpv_fpr, Bnpv_tpr,'go', label='NPV = {:0.3f}'.format(Bnpv))

        if min_err:
            plt.plot(1-mfpr, mtpr, 'bo', label='Error = {:0.3f}'.format(merr))

        if th_np:
            plt.plot(1-fpr_np, tpr_np, 'ko', label='Sp,Se = ({:0.3f},{:0.3f})'.format(1-fpr_np,tpr_np))

        if max_J:
            plt.plot(1-Jfpr, Jtpr, 'yo', label='Youden J = {:0.3f}'.format(Jval))

        plt.plot([0,1],[1,0],'k--')
        plt.xlim([-0.0,1.02])
        plt.ylim([0.0,1.02])
        plt.grid('on')
        plt.legend(loc='lower left')
        plt.ylabel('Sensitivity (TPR)')
        plt.xlabel('Specificity (TNR)')

    elif plot_type.lower() == 'ipr':
        # Inverse precision-recall. Plot Specificity = TNR v NPV
        tnr = 1-fpr
        fnr = 1-tpr
        npv = np.zeros(len(tnr))
        for i, tnr_val in enumerate(tnr):
            if tnr_val == 0.0:
                npv[i] = 0.0
            else:
                npv[i] = (tnr_val*Nn)/((tnr_val*Nn) + (fnr[i]*Np))

        plt.plot(tnr[:], npv[:], 'b',label='NPV-Specificity')
        plt.xlim([0.0,1.02])
        plt.ylim([0.0,1.02])
        plt.grid('on')
        plt.legend(loc='lower left')
        plt.ylabel('Negative Predictive Value (NPV)')
        plt.xlabel('Specificity (TNR)')

    elif plot_type.lower() == 'pr':
        # Plot PR-ROC TPR v PPV
        ppv = np.zeros(len(tpr))
        for i, tpr_val in enumerate(tpr):
            if tpr_val == 0.0:
                ppv[i] = 0.0
            else:
                ppv[i] = (tpr_val*Np)/((tpr_val*Np) + (fpr[i]*Nn))

        plt.plot(tpr[1:], ppv[1:], 'b',label='Precision-Recall')
        plt.xlim([0.0,1.02])
        plt.ylim([0.0,1.02])
        plt.grid('on')
        plt.legend(loc='lower left')
        plt.ylabel('Precision (PPV)')
        plt.xlabel('Recall (TPR)')

    else:
        # Plot FPR v TPR - ROC curve
        plt.plot(fpr, tpr, 'b', label='AUC = {:0.3f} +/-{:0.4f}'.format(roc_auc, sew))

        if ppv_npv:
            plt.plot(Bppv_fpr, Bppv_tpr,'ro', label='PPV = {:0.3f}'.format(Bppv))
            plt.plot(Bnpv_fpr, Bnpv_tpr,'go', label='NPV = {:0.3f}'.format(Bnpv))

        if min_err:
            plt.plot(mfpr, mtpr, 'bo', label='Error = {:0.3f}'.format(merr))

        if th_np:
            plt.plot(fpr_np, tpr_np, 'ko', label='FPR,TPR = ({:0.3f},{:0.3f})'.format(fpr_np,tpr_np))

        if max_J:
            plt.plot(Jfpr, Jtpr, 'yo', label='Youden J = {:0.3f}'.format(Jval))

        if plot_type.lower() == 'chi':
            xx, yy = np.mgrid[0:1:.01, 0:1:.01]
            grid = np.c_[xx.ravel(), yy.ravel()]
            chi = chi_sqr_val(grid[:,0],grid[:,1],1000,Np)
            chi = np.reshape(chi,xx.shape)
            cs = ax.contour(xx,yy,np.triu(chi),colors='k',
                            levels=[3.84,6.63,7.88,16,32,64,128,256,512,1024,2048],
                            linestyles='dotted',linewidths=0.5)
            plt.clabel(cs, fontsize=9, inline=1)
            title += ' (Chi-square Contours)'

        plt.plot([0,1],[0,1],'k--')
        plt.xlim([-0.02,1.0])
        plt.ylim([0.0,1.02])
        plt.grid('on')
        plt.legend(loc='lower right')
        plt.ylabel('True Positive Rate (TPR)')
        plt.xlabel('False Positive Rate (FPR)')

    plt.title(title)
    plt.show(fig)
    if save_pdf:
        fig.savefig(fname, bbox_inches='tight')

    return fig, ax
