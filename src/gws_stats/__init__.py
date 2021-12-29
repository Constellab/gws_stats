
# kruskalwallis
from .kruskalwallis.kruskalwallis import KruskalWallis
from .kruskalwallis.pairwise_kruskalwallis import PairwiseKruskalWallis

# mannwhitney
from .mannwhitney.mannwhitney import MannWhitney

# ttests
from .ttests.ttest1sample import TTestOneSample
from .ttests.ttest2sample_ind import TTestTwoIndepSamples
from .ttests.ttest2sample_rel import TTestTwoRelatedSamples

# anova
from .anova.pairwise_anova import PairwiseOneWayAnova
from .anova.anova import OneWayAnova

# wilcoxon
from .wilcoxon.wilcoxon import Wilcoxon
