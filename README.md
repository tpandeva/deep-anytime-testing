# deep-anytime-testing

1. Setup the environment and install packages
```
python3 -m venv my_env #  conda create --name my_env (if you use conda)
source my_env/bin/activate # conda activate my_env
pip install -r requirements.txt
```

2. Datasets
   
   **Blob Dataset**: The Blob dataset is a two-dimensional Gaussian mixture model with nine modes arranged on a 3 x 3 grid used by [[1]](#1) [[2]](#2) in their analysis.  The two distributions differ in their variance as visualized in the figure below.

    ![Blob Data](figures/blob_data.png)

   

### References
<a id="1">[1]</a>  Chwialkowski, K. P., Ramdas, A., Sejdinovic, D., and Gretton, A. (2015). Fast two-sample testing with
analytic representations of probability measures. Advances in Neural Information Processing Systems.

<a id="2">[2]</a>  Gretton, A., Borgwardt, K. M., Rasch, M. J., Schölkopf, B., and Smola, A. (2012). A kernel two-sample test.
The Journal of Machine Learning Research.
