# In-context Time Series Predictor

---

#### 1. Install Required Packages

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip3 install -r requirements.txt
```

#### 2. Download Other Datasets (Optional)

You can use the link provided by [Autoformer](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy) to download the datasets.

#### 3. Track the Training

```bash
nohup tensorboard --logdir runs --port 6006 --bind_all > tensorb.log 2>&1 &
```

#### 4. Run the Scripts

Run the training scripts under `./scripts` folder.
