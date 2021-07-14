# Qkeras Experiments
**Note**: The autoqkeras_tfdata_tfc files must run in Google Colab. They will fail in Vertex notebooks.
All the other files run in Vertex AI notebooks. Pick an instance with 30GB RAM min. to avoid memory problems. 
  
The _2gpus files implement:
* A pipeline using Tensorflow Data
* Parallel computation on 2GPUs if available with TF mirrored strategy. Note that this has been commented out in the code, because mirrored strategy in eager mode imposes a significant overhead (>50% in our example).

Given that the custom score function is not yet executable entirely on a Tensorflow graph, the benefits of such modifications are limited (about 3.5x faster).
The data files in those notebooks are stored locally.

The _tfdata_tfc files implement:
* A Tensorflow data pipeline
* Hyperparameter tuning on Vizier service using the Tensorflow Cloud library. Note that for now this does not use a custom objective, but a simple valuation loss. It is still work in progress.

Due to limitations in Tensorflow Cloud, they must run in a google Colab environment. Run the setup notebook first, then the main one.
The data in this case must reside on in a google storage bucket and the parameter search is run as a series of AI Platform jobs.