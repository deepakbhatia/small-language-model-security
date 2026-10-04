# Train SEI stage-1 on Kaggle (free GPU)

1. Create a new **Notebook** on Kaggle.
2. Settings:
   - Accelerator: **GPU T4** (or available free GPU)
   - Internet: **On**
3. **Add Input** → Microsoft GUIDE dataset  
   (`Microsoft/microsoft-security-incident-prediction` or your local copy of that dataset).
4. Copy cells from [`notebooks/kaggle_train_stage1.ipynb`](../notebooks/kaggle_train_stage1.ipynb) **or** clone this repo in the first cell.
5. Download adapters from `/kaggle/working/slms/checkpoints/` when finished.

## Clone pattern

```python
!git clone https://github.com/<YOUR_USER>/slms.git
%cd /kaggle/working/slms
```

Replace `<YOUR_USER>` with your GitHub username/org after you push this repo.
