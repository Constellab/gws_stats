
# anova
from .anova.anova import OneWayAnova
from .anova.pairwise_anova import PairwiseOneWayAnova
# kruskalwallis
from .kruskalwallis.kruskalwallis import KruskalWallis
from .kruskalwallis.pairwise_kruskalwallis import PairwiseKruskalWallis
# mannwhitney
from .mannwhitney.mannwhitney import MannWhitney
# normaltest
from .normaltest.normaltest import NormalTest
# ttests
from .ttests.ttest1sample import TTestOneSample
from .ttests.ttest2sample_ind import TTestTwoIndepSamples
from .ttests.ttest2sample_rel import TTestTwoRelatedSamples
# wilcoxon
from .wilcoxon.wilcoxon import Wilcoxon
# correlation
from .correlation.correlation import PairwiseCorrelationCoef
