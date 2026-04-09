# GENIE: Watermarking Graph Neural Networks for Link Prediction

### Introduction
The rapid adoption, usefulness, and resource-intensive training of Graph Neural Network (GNN) models have made them an invaluable intellectual property in graph-based machine learning. However, their wide-spread adoption also makes them susceptible to stealing, necessitating robust Ownership Demonstration (OD) techniques. Watermarking is a promising OD framework for deep neural networks, but existing methods fail to generalize to GNNs due to the non-Euclidean nature of graph data. Existing works on GNN watermarking primarily focus on node and graph classification, overlooking Link Prediction (LP). In this paper, we propose Genie (watermarking Graph nEural Networks for lInk prEdiction), the first scheme to watermark GNNs for LP. Genie creates a novel backdoor for both node-representation and subgraph-based LP methods, utilizing a unique trigger set and a secret watermark vector. Our OD scheme is equipped with Dynamic Watermark Thresholding (DWT), ensuring high verification probability while addressing practical issues in existing OD schemes. We extensively evaluate Genie across 4 diverse model architectures (i.e., SEAL, GCN, GraphSAGE and NeoGNN), 7 real-world datasets and 21 watermark removal techniques and demonstrate its robustness to watermark removal and ownership piracy attacks. Finally, we discuss adaptive attacks against Genie and a defense strategy to counter it. The codebase and related artifacts are publicly available at our Project Page.

### Please don't forget to cite our paper.
Venkata Sai Pranav Bachina, Aaryan Ajay Sharma, Charu Sharma, Ankit Gangwal. <br>
GENIE: Watermarking Graph Neural Networks for Link Prediction. <br>
In Transactions on Machine Learning Research (TMLR), 2026.<br>

### People 
1. <a href="https://bachina-pranav.github.io/">Venkata Sai Pranav Bachina</a>, International Institute of Information Technology, Hyderabad, India<br/>
2. <a href="https://aaryanajaysharma.github.io/">Aaryan Ajay Sharma</a>, International Institute of Information Technology, Hyderabad, India<br/>
3. <a href="https://charusharma.org/">Charu Sharma</a>, International Institute of Information Technology, Hyderabad, India<br/>
4. <a href="https://ciaoankit.github.io/">Ankit Gangwal</a>, International Institute of Information Technology, Hyderabad, India<br/>
