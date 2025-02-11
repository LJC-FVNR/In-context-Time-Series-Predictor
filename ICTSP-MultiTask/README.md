## ICTSP - Multi-task Time Series Foundation Model Pretraining

### 1. Install Required Packages

To get started, install the necessary dependencies:

```bash
pip3 install torch
pip3 install -r requirements.txt
```

### 2. Manage Your Pretraining Data

The `ts_generation` folder contains two time series generators:

- `TS_DS_Generator.py`
- `Ode_generator.py`

These can generate multivariate time series data with inter-series dependencies. Alternatively, you can use your own custom data. To do so, place your data files (in `.csv` format by default) in a new folder under `dataset/`. Then update the pretraining configuration file to include your new data as follow.

### 3. Edit the Training Configuration

Customize the training settings by editing the file `configs/pretrain_configs_sequential.json`.

#### Adjust Data Source Weights

To balance training across different data sources, assign weights to the datasets. Use the first-level folder name under `dataset/` as the data source name. For example, the following configuration sets equal-weight sampling for the data in `dataset/ICL_pretrain_1/` and `dataset/ODE_pretrain_1/`:

```json
"source_weight_reg": {
    "ICL_pretrain_1": 1,
    "ODE_pretrain_1": 1
}
```

#### Configure Task Weights

Enable multi-task time series pretraining by setting weights for each task. For instance:

```json
"token_type_weight_reg": {
    "forecasting": 1,
    "classification": 0,
    "imitation": 0,
    "imputation": 0.2,
    "cropping": 0.2,
    "reflection": 0.2,
    "shifting": 0.2,
    "hyperres": 0.2,
    "statistics": 0,
    "differencing": 0.2,
    "movingavg": 0.2,
    "expsmoothing": 0,
    "decomposition": 0
}
```

### 4. Run the Scripts

After configuring `configs/pretrain_configs_sequential.json`, start the pretraining process by running the script:

```bash
./icpretrain.sh
```

This script uses a single time series data file as the validation and test reference, specified as `data_name="ETTh2.csv"`.

To fine-tune the model on a specific dataset with a fixed (lookback, future) pair after pretraining, run:

```bash
./icfinetune.sh
```

Modify `data_name="ETTh2.csv"` to point to your dataset path.

For fine-tuning on multiple datasets with flexible (lookback, future) settings, place your datasets in a new folder under `dataset_finetune/`. Then, change the following setting in `configs/pretrain_configs_finetune.json`:

```json
"source_weight_reg": {
        "ETTh2": 1
    }
...
"use_legacy_dataloader": false,
```

This enables the ICTSP tokenizer and dataloader for multi-task training.

### 5. Track Your Training

To monitor the training process, use TensorBoard:

```bash
nohup tensorboard --logdir runs --port 6006 --bind_all > tensorb.log 2>&1 &
```

