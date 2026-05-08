# Spatial Transcriptomics Prediction from H&E Images

This repository implements spatial transcriptomics gene expression prediction from H&E whole slide images using different foundation models and spatial neighbor aggregation techniques, built on top of the [HEST-Bench](https://github.com/mahmoodlab/HEST) framework.
                                                                                                                                      
## Task                                                   

I use frozen patch-level embeddings from foundation models to predict RNA gene expression at each spatial spot on a tissue slide. For each spot, an image patch is extracted from the H&E whole slide image, encoded by a foundation model, and passed to a ridge regression probe that predicts the expression values of 50 highly variable genes.                                        
                                                                                                                                      
## Settings                                               

**RNA expression values (target):** Raw gene counts are normalized using `log1p` (i.e., `log(count + 1)`). This is standard practice in single-cell and spatial transcriptomics analysis because raw counts are highly zero-inflated and right-skewed — a small number of genes have very high counts while most have near-zero counts. `log1p` compresses this dynamic range, stabilizes variance, and makes the target distribution more suitable for linear regression.

**Evaluation metric:** Pearson correlation coefficient, computed per gene between predicted and actual expression values across all test spots, then averaged across all 50 genes. Pearson correlation is preferred over MSE because it is scale-invariant — it measures whether the model correctly ranks spots by expression level, regardless of absolute scale. This makes it robust to normalization differences across slides and datasets.                   

**K-fold validation:** Each dataset is evaluated using K-fold cross-validation (depending on the case). Results are reported as the mean and standard deviation of the per-fold Pearson correlations. This accounts for variability introduced by the choice of train/test split and gives a more reliable estimate of generalization performance.                                                  
                                                            
**Dimensionality reduction:** By default, PCA is applied to reduce embeddings to 256 dimensions before ridge regression. This ensure fair comparison between models.
                                                                                                                                      
## Environment Setup                                      

Refer to the official [HEST GitHub repository](https://github.com/mahmoodlab/HEST.git) for full documentation.                      
  
```bash                                                                                                                             
conda create -n hest python=3.11                          
conda activate hest
pip install -e .
pip install -e ".[benchmark]"

If a GPU is available, install GPU-accelerated libraries for faster inference:                                                      
  
pip install \
    --extra-index-url=https://pypi.nvidia.com \
    cudf-cu12==24.6.* dask-cudf-cu12==24.6.* cucim-cu12==24.6.* \
    raft-dask-cu12==24.6.*                                                                                                    
```

DINOv3 weights must be downloaded separately and placed at a local path before running.                                             
                                                            
Script Launching                                                                                                                    
                                                            
Foundation Models (Baseline)

```bash 
python implementations/benchmark.py \
  --model_name uni_v2 \
  --bench_data_root /PATH/TO/DATA \
  --embed_dataroot /PATH/TO/EMBEDDINGS \                                                                                            
  --results_dir /PATH/TO/RESULTS
```
                                                                                                                                      ┌───────────────────┬──────────────────────────────────────────────────────────────────────┐
│     Argument      │                              Description                               │
├───────────────────┼────────────────────────────────────────────────────────────────────────┤                                      
│ --model_name      │ Foundation model to use. One of uni_v2, dinov3_vitl16, dinov3_vitb16   │
├───────────────────┼────────────────────────────────────────────────────────────────────────┤                                      
│ --bench_data_root │ Root directory containing the downloaded HEST-Bench datasets           │                                      
├───────────────────┼────────────────────────────────────────────────────────────────────────┤
│ --embed_dataroot  │ Directory where extracted patch embeddings will be saved               │                                      
├───────────────────┼────────────────────────────────────────────────────────────────────────┤                                      
│ --results_dir     │ Directory where prediction results will be saved                       │
├───────────────────┼────────────────────────────────────────────────────────────────────────┤                                      
│ --weights         │ (DINOv3 only) Path to the local DINOv3 pretrained weights file         │
├───────────────────┼────────────────────────────────────────────────────────────────────────┤                                      
│ --batch_size      │ Batch size for embedding extraction (default: 128)                     │
├───────────────────┼────────────────────────────────────────────────────────────────────────┤                                      
│ --num_workers     │ Number of dataloader workers (default: 1)                              │
├───────────────────┼────────────────────────────────────────────────────────────────────────┤                                      
│ --dimreduce       │ Dimensionality reduction before regression. PCA or None (default: PCA) │
├───────────────────┼────────────────────────────────────────────────────────────────────────┤                                      
│ --latent_dim      │ PCA output dimension (default: 256)                                    │
├───────────────────┼────────────────────────────────────────────────────────────────────────┤                                      
│ --seed            │ Random seed for reproducibility (default: 1)                           │
└───────────────────┴────────────────────────────────────────────────────────────────────────┘                                      
                                                            
Example for DINOv3:                                                                                                                 
                                                            
python implementations/benchmark.py \
  --model_name dinov3_vitb16 \
  --weights /PATH/TO/dinov3_vitb16_weights.pth \
  --bench_data_root /PATH/TO/DATA \                                                                                                 
  --embed_dataroot /PATH/TO/EMBEDDINGS \                                                                                            
  --results_dir /PATH/TO/RESULTS                                                                                                    
                                                                                                                                      
Foundation Models + Spatial Neighbor Aggregation          

python implementations/neighbor_agg.py \
  --model_name dinov3_vitb16 \                                                                                                      
  --weights /PATH/TO/dinov3_vitb16_weights.pth \
  --spatial_agg \                                                                                                                   
  --spatial_k 8 \                                         
  --spatial_agg_mode concat \                                                                                                       
  --bench_data_root /PATH/TO/DATA \
  --embed_dataroot /PATH/TO/EMBEDDINGS \                                                                                            
  --results_dir /PATH/TO/RESULTS                          
                                                                                                                                      
Additional arguments for neighbor aggregation:
                                                                                                                                      
┌────────────────────┬───────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│      Argument      │                                                Description                                                │
├────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ --spatial_agg      │ Flag to enable spatial neighbor aggregation                                                               │
├────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ --spatial_k        │ Number of nearest neighbor spots to aggregate. 8 = 3×3 window minus center, 24 = 5×5 window minus center  │  
├────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────┤  
│ --spatial_agg_mode │ Aggregation mode: concat (concatenate neighbor embeddings, then PCA) or mean (average all embeddings,     │  
│                    │ preserves dimensionality)                                                                                 │  
└────────────────────┴───────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                                                                                                      
Results                                                   

Mean Pearson correlation across 9 HEST-Bench datasets (IDC, PRAD, PAAD, SKCM, COAD, READ, CCRCC, LUNG, LYMPH_IDC):                  
  
┌───────────────────────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┬───────────┬────────┐          
│          Method           │  IDC   │  PRAD  │  PAAD  │  SKCM  │  COAD  │  READ  │ CCRCC  │  LUNG  │ LYMPH_IDC │  Avg   │
├───────────────────────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┼───────────┼────────┤          
│ UNI v2                    │ 0.5898 │ 0.3567 │ 0.5000 │ 0.6611 │ 0.3018 │ 0.2226 │ 0.2640 │ 0.5589 │ 0.2728    │ 0.4142 │
├───────────────────────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┼───────────┼────────┤          
│ DINOv3-L16                │ 0.4911 │ 0.3712 │ 0.3738 │ 0.5474 │ 0.2511 │ 0.1259 │ 0.2172 │ 0.5025 │ 0.2448    │ 0.3472 │          
├───────────────────────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┼───────────┼────────┤          
│ DINOv3-B16                │ 0.4811 │ 0.3127 │ 0.3956 │ 0.5057 │ 0.2130 │ 0.1058 │ 0.2191 │ 0.5002 │ 0.2453    │ 0.3309 │          
├───────────────────────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┼───────────┼────────┤          
│ DINOv3-B16 + Concat (k=8) │ 0.4826 │ 0.3529 │ 0.3897 │ 0.4877 │ 0.2648 │ 0.0905 │ 0.2222 │ 0.5049 │ 0.2404    │ 0.3373 │
├───────────────────────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┼───────────┼────────┤          
│ DINOv3-B16 + Mean (k=8)   │ 0.4476 │ 0.2230 │ 0.3420 │ 0.4195 │ 0.1491 │ 0.0973 │ 0.2528 │ 0.4008 │ 0.2410    │ 0.2859 │
└───────────────────────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┴───────────┴────────┘          
                                                            
Summary                                                                                                                             
                                                            
UNI v2 achieves the highest average Pearson correlation (0.4142), outperforming both DINOv3 variants by a substantial margin. This is expected: UNI v2 is pretrained specifically on histopathology images, making its embeddings more aligned with tissue-level biological features relevant to gene expression.                                                                                    

Spatial neighbor aggregation with concatenation (k=8) provides a modest and inconsistent improvement over the DINOv3-B16 baseline (+0.006 average, wins 5/9 datasets), with the largest gains on PRAD (+0.040) and COAD (+0.052) — cancer types where local tissue microenvironment may play a stronger role in expression patterns. Mean aggregation consistently hurts performance (8/9 datasets, −0.045 average), likely because averaging over 9 spots over-smooths the embedding and dilutes the discriminative signal of the center spot.

Limitations

- Spatial aggregation did not consistently improve performance. The benefit of concat aggregation was marginal and                  
dataset-dependent, suggesting that k=8 pixel-space nearest neighbors may not reliably capture biologically meaningful tissue context
for all cancer types. A learnable aggregation head (e.g., a small transformer) could better weight neighbor contributions, but     
  would require more training data to avoid overfitting.    
  - Mean aggregation is not suitable for this task. Averaging across all neighboring embeddings collapses the individual spot's
  identity into its microenvironment, which loses the local discriminative signal that ridge regression relies on.                    
  - The domain gap between general and pathology-specific encoders is large. DINOv3 models, despite being large-scale vision
  transformers, trail UNI v2 by ~8 Pearson points on average. Spatial aggregation alone is insufficient to close this gap, reinforcing
   that domain-specific pretraining on histopathology data is the primary driver of performance on ST prediction tasks.
  - Limited training data per dataset. Each HEST-Bench dataset contains ~3,000–5,000 spatial spots per slide, constraining how much   
  the regression probe can learn. More expressive aggregation strategies (attention, GNNs) would likely overfit in this regime without
   additional regularization or more slides.
