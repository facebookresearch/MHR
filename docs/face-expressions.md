# Facial expression mapping

MHR v1.x uses 72 artist-sculpted, sparse semantic facial blendshapes that follow
the Facial Action Coding System (FACS). They are not PCA components, and not
every control has a one-to-one correspondence with a canonical FACS Action
Unit. For example, the space includes unilateral controls, eye gaze, jaw
direction, and individual lip quadrants. The mapping is identical for LODs 0
through 6.

The zero-based index in this table is the column used by
`face_expr_coeffs[:, index]`. The same coefficient maps to the generic FBX shape
`shape_{45 + index}` and PyMomentum parameter `blend_{45 + index}`. `_L` and
`_R` refer to the character's left and right; lip-quadrant suffixes additionally
use `T` for top and `B` for bottom.

The names are also available programmatically:

```python
import torch

from mhr import FACE_EXPRESSION_NAMES

face_expr_coeffs = torch.zeros(1, len(FACE_EXPRESSION_NAMES))
jaw_drop_index = FACE_EXPRESSION_NAMES.index("jawDrop")
face_expr_coeffs[:, jaw_drop_index] = 1.0
```

The mapping is package-level so it can also be used with the shipped v1.x
TorchScript model, which exposes the expression count but not semantic names.

| Index | Semantic name | FBX / PyMomentum name |
| ---: | --- | --- |
| 0 | `browLowerer_L` | `shape_45` / `blend_45` |
| 1 | `browLowerer_R` | `shape_46` / `blend_46` |
| 2 | `cheekPuff_L` | `shape_47` / `blend_47` |
| 3 | `cheekPuff_R` | `shape_48` / `blend_48` |
| 4 | `cheekRaiser_L` | `shape_49` / `blend_49` |
| 5 | `cheekRaiser_R` | `shape_50` / `blend_50` |
| 6 | `cheekSuck_L` | `shape_51` / `blend_51` |
| 7 | `cheekSuck_R` | `shape_52` / `blend_52` |
| 8 | `chinRaiser_B` | `shape_53` / `blend_53` |
| 9 | `chinRaiser_T` | `shape_54` / `blend_54` |
| 10 | `dimpler_L` | `shape_55` / `blend_55` |
| 11 | `dimpler_R` | `shape_56` / `blend_56` |
| 12 | `eyesClosed_L` | `shape_57` / `blend_57` |
| 13 | `eyesClosed_R` | `shape_58` / `blend_58` |
| 14 | `eyesLookDown_L` | `shape_59` / `blend_59` |
| 15 | `eyesLookDown_R` | `shape_60` / `blend_60` |
| 16 | `eyesLookLeft_L` | `shape_61` / `blend_61` |
| 17 | `eyesLookLeft_R` | `shape_62` / `blend_62` |
| 18 | `eyesLookRight_L` | `shape_63` / `blend_63` |
| 19 | `eyesLookRight_R` | `shape_64` / `blend_64` |
| 20 | `eyesLookUp_L` | `shape_65` / `blend_65` |
| 21 | `eyesLookUp_R` | `shape_66` / `blend_66` |
| 22 | `innerBrowRaiser_L` | `shape_67` / `blend_67` |
| 23 | `innerBrowRaiser_R` | `shape_68` / `blend_68` |
| 24 | `jawDrop` | `shape_69` / `blend_69` |
| 25 | `jawSidewaysLeft` | `shape_70` / `blend_70` |
| 26 | `jawSidewaysRight` | `shape_71` / `blend_71` |
| 27 | `jawThrust` | `shape_72` / `blend_72` |
| 28 | `lidTightener_L` | `shape_73` / `blend_73` |
| 29 | `lidTightener_R` | `shape_74` / `blend_74` |
| 30 | `lipCornerDepressor_L` | `shape_75` / `blend_75` |
| 31 | `lipCornerDepressor_R` | `shape_76` / `blend_76` |
| 32 | `lipCornerPuller_L` | `shape_77` / `blend_77` |
| 33 | `lipCornerPuller_R` | `shape_78` / `blend_78` |
| 34 | `lipFunneler_LB` | `shape_79` / `blend_79` |
| 35 | `lipFunneler_LT` | `shape_80` / `blend_80` |
| 36 | `lipFunneler_RB` | `shape_81` / `blend_81` |
| 37 | `lipFunneler_RT` | `shape_82` / `blend_82` |
| 38 | `lipPressor_L` | `shape_83` / `blend_83` |
| 39 | `lipPressor_R` | `shape_84` / `blend_84` |
| 40 | `lipPucker_L` | `shape_85` / `blend_85` |
| 41 | `lipPucker_R` | `shape_86` / `blend_86` |
| 42 | `lipStretcher_L` | `shape_87` / `blend_87` |
| 43 | `lipStretcher_R` | `shape_88` / `blend_88` |
| 44 | `lipSuck_LB` | `shape_89` / `blend_89` |
| 45 | `lipSuck_LT` | `shape_90` / `blend_90` |
| 46 | `lipSuck_RB` | `shape_91` / `blend_91` |
| 47 | `lipSuck_RT` | `shape_92` / `blend_92` |
| 48 | `lipTightener_L` | `shape_93` / `blend_93` |
| 49 | `lipTightener_R` | `shape_94` / `blend_94` |
| 50 | `lipsToward_LB` | `shape_95` / `blend_95` |
| 51 | `lipsToward_LT` | `shape_96` / `blend_96` |
| 52 | `lipsToward_RB` | `shape_97` / `blend_97` |
| 53 | `lipsToward_RT` | `shape_98` / `blend_98` |
| 54 | `lowerLipDepressor_L` | `shape_99` / `blend_99` |
| 55 | `lowerLipDepressor_R` | `shape_100` / `blend_100` |
| 56 | `mouthLeft` | `shape_101` / `blend_101` |
| 57 | `mouthRight` | `shape_102` / `blend_102` |
| 58 | `nasolabialFurrow_L` | `shape_103` / `blend_103` |
| 59 | `nasolabialFurrow_R` | `shape_104` / `blend_104` |
| 60 | `noseWrinkler_L` | `shape_105` / `blend_105` |
| 61 | `noseWrinkler_R` | `shape_106` / `blend_106` |
| 62 | `nostrilCompressor_L` | `shape_107` / `blend_107` |
| 63 | `nostrilCompressor_R` | `shape_108` / `blend_108` |
| 64 | `nostrilDilator_L` | `shape_109` / `blend_109` |
| 65 | `nostrilDilator_R` | `shape_110` / `blend_110` |
| 66 | `outerBrowRaiser_L` | `shape_111` / `blend_111` |
| 67 | `outerBrowRaiser_R` | `shape_112` / `blend_112` |
| 68 | `upperLidRaiser_L` | `shape_113` / `blend_113` |
| 69 | `upperLidRaiser_R` | `shape_114` / `blend_114` |
| 70 | `upperLipRaiser_L` | `shape_115` / `blend_115` |
| 71 | `upperLipRaiser_R` | `shape_116` / `blend_116` |
