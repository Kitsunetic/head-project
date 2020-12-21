# Head Orientation Prediction

![Snapshot](img/Snapshot.PNG)

![BarGraph](img/BarGraph.PNG)

![FinalResult](img/FinalResult.PNG)

# Docs

[Final Presentation](Final%20Report.pdf)

# Usage

```bash
# Experiment directory will be automatically generated in the results folder.
# "results/0000-CRNNC_Hardswish-win_120-epoch_200-batch_size_256[-comment]"  
python run_training.py \
  --window-size 120 \
  --epochs 200 \
  --batch-size 256 \
  --cpus 8 \
  --network CRNNC_Hardswish \
  --dataset data/1116 \
  --result results \
  --comment {any comment or not}
```
