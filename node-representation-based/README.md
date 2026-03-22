This repository contains all the watermarked models and their corresponding watermark datasets for GCN and SAGE models on all datasets. To check the performance of GCN,SAGE on test dataset and watermarked dataset for all datasets(NS,celegans etc.), run the following code:  

```shell
chmod +x main.sh
./main.sh
```

To reproduce the results of Table 10. (Impact of Weight Quantization on the watermark), run the following code:

```shell
chmod +x run_wq.sh
./run_wq.sh
```