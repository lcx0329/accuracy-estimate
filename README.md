# accuracy-estimate

## Quick Experiments

### project structure

This project contains the following main components, each with a specific function:

```markdown
├── calibration                     # Toolkit for calibration experiments
├── models_helg                     # Guidance for preparing models
├── conf                            # Project configuration files with basic settings
├── toolkit                         # Custom toolkit for auxiliary functions in experiments
├── main.py                         # Main code file for reproducing Aries and ATC experiments, supports different uncertainty metrics
├── temperature.py                  # Code for temperature calibration experiments
├── threshold.py                    # Code for ATC threshold experiments
├── bucket.py                       # Code for experiments on the number of buckets in Aries
```

## Example

### Using ImageNet-200 Dataset and ResNet101 Model

If you wish to conduct experiments using the ImageNet-200 dataset and ResNet101 model, focusing on Entropy as the uncertainty metric, you can follow these steps:

1. Ensure you have downloaded and prepared the ImageNet-200 dataset.
2. Ensure the ResNet101 model is available for your environment.
3. Run the experiment using the following command:

```sh
python main.py -data ImageNet-200 -net resnet101 -save_path ./result/entropy.csv  -index entropy
```

This command will start an experiment using the ResNet101 model on the ImageNet-200 dataset, employing Entropy as the uncertainty metric. The results will be saved in `./result/entropy.csv`.