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

3. Structure
train.py contains the training pipeline for each experiment. Here, all objects are initialized by making use of hydra's config files (see folder configs) and the training is performed. The training pipeline consists of the following steps:
     - **Initialize the data generator** (e.g. Blob dataset) This class yields samples from two distributions. The output object is of class torch.Dataset. Independently on the dataset there are two parameters that should be specified
       - samples: number of samples to be generated
       - type: type of the experiment "type2" (alternative holds) "type11" (null holds and the data comes from the first class), "type12" (null holds and the data comes from the second class)
        
       The configuration files for each data generator are in folder configs/data, e.g. the file blob.yaml contains
       ```
       _target_: "data.blob.BlobDataGen"
       samples: 1000 # mandatory parameter
       type: "type2" # mandatory parameter
       r: 3 # dataset specific parameter
       d: 2 # dataset specific parameter
       rho: 0.03 # dataset specific parameter
       with_labels: false # dataset specific parameter
       ``` 
     - **Initialize the operator.** An object of class Operator (operators/base/Operator). So far we have implemented 
       - the symmetry operator: $\tau(x) = -x$
       - swap operator: $\tau(x,y) = (y,x)$ and $\tau((x_1,y_1),(x_2, y_2)) = ((x_1,y_2),(x_2, y_1))$. As for the data generator the default initialization parameters for the SwapOperator (operators/swap/SwapOperator) are stored in a hydra config file: configs/operator/swap.yaml
         ```
         _target_: "operators.swap.SwapOperator"
         p: 3 # tau input dimension (e.g. the number of features in either x or y in the two-sample test or the total number of feature is x_1 and y_1 ( or x_2 and y_2))
         d: 2 # the starting index for swapping (e.g. here we swap the last feature of the 3D input) 
         ``` 
     - **Initialize the model.** The first model is a simple MLP with number and dimensions of hidden layers specified by the user. See the corresponding config files model/mlp.yaml for the default parameters. To build an MLP with four hidden layers with size 40 the user should specify hidden_layer_size: [40, 40, 40, 40].

     - **Initialize and perform the training** This step initializes a trainer object of class trainer.trainer.Trainer with configuration in the default configuration file (configs/config.yaml)
   
   #### **Important:** The user can specify all needed parameters (i.e. overwrite the defaults) in the configs/config.yaml file!

4. Experiments

    Run two sample test for Blob dataset with the following command. The number of dimensions is 2 and thus operator.p=2. For the two-sample test we always use operator.d=0. The model input is 2 x 2 = 4.
    ```
    python train.py data=blob data.type="type2" data.samples=900 data.with_labels=false operator=swap operator.p=2 operator.d=0 model=mlp model.input_size=4
    # type I error experiments:
    python train.py data=blob data.type="type12" data.samples=900 data.with_labels=false operator=swap operator.p=2 operator.d=0 model=mlp model.input_size=4
    python train.py data=blob data.type="type11" data.samples=900 data.with_labels=false operator=swap operator.p=2 operator.d=0 model=mlp model.input_size=4
    ```

    Run independence test for Blob dataset. Here the null is that the data set X is independent of the class labels Y. We can get the labels by setting data.with_labels=true. The data generator samples data $(X,Y)\in \mathbb{R}^3$ and thus operator.p=3. The swap operator swaps the last feature of the input and thus operator.d=2. The model input is 2 x 3 = 6. 
    ``` 
    python train.py data=blob data.type="type2" data.samples=900 data.with_labels=true operator=swap operator.p=3 operator.d=2 model=mlp model.input_size=6
    ```

### References
<a id="1">[1]</a>  Chwialkowski, K. P., Ramdas, A., Sejdinovic, D., and Gretton, A. (2015). Fast two-sample testing with
analytic representations of probability measures. Advances in Neural Information Processing Systems.

<a id="2">[2]</a>  Gretton, A., Borgwardt, K. M., Rasch, M. J., Schölkopf, B., and Smola, A. (2012). A kernel two-sample test.
The Journal of Machine Learning Research.
