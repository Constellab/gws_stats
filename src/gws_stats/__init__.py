
# anova
from .anova.anova import OneWayAnova
from .anova.pairwise_anova import PairwiseOneWayAnova
# correlation
from .correlation.pearson import PearsonCorrelation
from .correlation.spearman import SpearmanCorrelation
# kruskalwallis
from .kruskalwallis.kruskalwallis import KruskalWallis
from .kruskalwallis.pairwise_kruskalwallis import PairwiseKruskalWallis
# mannwhitney
from .mannwhitney.mannwhitney import MannWhitney
# normaltest
from .normaltest.normaltest import NormalTest
# pvalue adjust
from .pval_adjust.pval_adjust import PValueAdjust
# ttests
from .ttests.ttest1sample import TTestOneSample
from .ttests.ttest2sample_ind import TTestTwoIndepSamples
from .ttests.ttest2sample_rel import TTestTwoRelatedSamples
# wilcoxon
from .wilcoxon.wilcoxon import Wilcoxon
