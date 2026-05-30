from .swin3d import (
    Swin3DConfig,
    Swin3DImageProcessor,
    Swin3DForVideoClassification
)
from .video_resnet import (
    VideoResNetConfig,
    VideoResNetImageProcessor,
    VideoResNetForVideoClassification
)
from .s3d import (
    S3DConfig,
    S3DImageProcessor,
    S3DForVideoClassification
)
from .mvit import (
    MViTConfig,
    MViTImageProcessor,
    MViTForVideoClassification
)
from .sl_gcn import (
    SLGCNConfig,
    SLGCNFeatureExtractor,
    SLGCNForGraphClassification
)
from .videomae import (
    VideoMAEConfig,
    VideoMAEImageProcessor,
    VideoMAEForVideoClassification
)
from .spoter import (
    SPOTERConfig,
    SPOTERFeatureExtractor,
    SPOTERForGraphClassification
)
from .dsta_slr import (
    DSTASLRConfig,
    DSTASLRFeatureExtractor,
    DSTASLRForGraphClassification
)
