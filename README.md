# COMP7860
Course project

Steps: 
1. Download loc-gowalla_totalCheckins.txt.gz from 
https://snap.stanford.edu/data/loc-gowalla.html

2. Adjust values KDTSearchHeight and epsilon in kdTree_DP_WithTime.py, and then run it to generate DP datasets: 
>python kdTree_DP_WithTime.py

References: 

[1] The code for kdTree is taken from: 
https://www.astroml.org/book_figures/chapter2/fig_kdtree_example.html

[2]For measuring the data quality of DP, I used the LSTM predictor from: 
https://github.com/xiaochus/TrafficFlowPrediction.git
