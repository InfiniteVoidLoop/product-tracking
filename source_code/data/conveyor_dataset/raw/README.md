# Raw Pipeline Object Dataset

Auto-sampled from short production/conveyor videos.

- YOLO class 0: `product`
- Saved frames: 9
- Saved boxes: 9
- Confidence threshold: 0.3
- Sample interval: 0.5 seconds

Review labels before using this as final training data. These are
auto-labels from the selected YOLO model, mapped into one product class.

To split this into train/val/test for the project:

```bash
python scripts/prepare_dataset.py \
  --source data/raw_pipeline_objects \
  --output data/pipeline_objects \
  --class-names product
```
