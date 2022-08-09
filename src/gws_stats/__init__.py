
# anova
from .anova.anova import OneWayAnova
# correlation
from .correlation.pearson import PearsonCorrelation
from .correlation.spearman import SpearmanCorrelation
# kruskalwallis
from .kruskalwallis.kruskalwallis import KruskalWallis
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
