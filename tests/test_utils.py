from waffle_hub.schema.evaluate import (
    ClassificationMetric,
    InstanceSegmentationMetric,
    ObjectDetectionMetric,
)
from waffle_hub.schema.fields import Annotation
from waffle_hub.utils.evaluate import (
    evaluate_classification,
    evaluate_object_detection,
    evaluate_segmentation,
)


def test_evaluate_classification():
    result: ClassificationMetric = evaluate_classification(
        preds=[
            [Annotation.classification(category_id=1)],
            [Annotation.classification(category_id=2)],
            [Annotation.classification(category_id=1)],
        ],
        labels=[
            [Annotation.classification(category_id=1)],
            [Annotation.classification(category_id=2)],
            [Annotation.classification(category_id=3)],
        ],
        num_classes=3,
    )

    assert abs(result.accuracy - 2 / 3) < 1e-5


def test_evaluate_object_detection():
    result: ObjectDetectionMetric = evaluate_object_detection(
        preds=[
            [
                Annotation.object_detection(category_id=1, bbox=[0, 0, 1, 1], score=1.0),
                Annotation.object_detection(category_id=2, bbox=[0, 0, 1, 1], score=1.0),
                Annotation.object_detection(category_id=3, bbox=[0, 0, 1, 1], score=1.0),
            ],
            [
                Annotation.object_detection(category_id=1, bbox=[0, 0, 1, 1], score=1.0),
                Annotation.object_detection(category_id=2, bbox=[0, 0, 1, 1], score=1.0),
                Annotation.object_detection(category_id=2, bbox=[0, 0, 1, 1], score=1.0),
            ],
            [
                Annotation.object_detection(category_id=1, bbox=[0, 0, 1, 1], score=1.0),
                Annotation.object_detection(category_id=1, bbox=[0, 0, 1, 1], score=1.0),
                Annotation.object_detection(category_id=1, bbox=[0, 0, 1, 1], score=1.0),
            ],
        ],
        labels=[
            [
                Annotation.object_detection(category_id=1, bbox=[0, 0, 1, 1], score=1.0),
                Annotation.object_detection(category_id=2, bbox=[0, 0, 1, 1], score=1.0),
                Annotation.object_detection(category_id=3, bbox=[0, 0, 1, 1], score=1.0),
            ],
            [
                Annotation.object_detection(category_id=1, bbox=[0, 0, 1, 1], score=1.0),
                Annotation.object_detection(category_id=2, bbox=[0, 0, 1, 1], score=1.0),
                Annotation.object_detection(category_id=3, bbox=[0, 0, 1, 1], score=1.0),
            ],
            [
                Annotation.object_detection(category_id=1, bbox=[0, 0, 1, 1], score=1.0),
                Annotation.object_detection(category_id=2, bbox=[0, 0, 1, 1], score=1.0),
                Annotation.object_detection(category_id=3, bbox=[0, 0, 1, 1], score=1.0),
            ],
        ],
        num_classes=3,
    )

    assert result.mAP < 1.0
