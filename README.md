# Qkeras Experiments
**Note**: Autoqkeras_tfdata_tfc_setup and the Qkeras_tfdata_tfc_consolidated must run in Google Colab. They will fail in Vertex notebooks.
All the other files run in Vertex AI notebooks. Pick an instance with 30GB RAM min. to avoid memory problems. 
  
The *_2gpus* files implement:
* A pipeline using Tensorflow Data
* Parallel computation on 2GPUs if available with TF mirrored strategy. Note that this has been commented out in the code, because mirrored strategy in eager mode imposes a significant overhead (>50% in our example).


The *tfdata_tfc* files implement:
* A Tensorflow data pipeline
* A custom metric as a Keras callback
* Hyperparameter tuning on Vizier service using the Tensorflow Cloud library. Custom metric is reported to the tuner.  

Note that the hyperparameter tuning service for now supports only 1 metric. The Vizier optimization service (beta) is actually capable of multiple objectives.

Due to limitations in Tensorflow Cloud, they must run in a google Colab environment. Run the setup notebook first, then the "consolidated" one.
The data in this case must reside on in a google storage bucket and the parameter search is run as a series of AI Platform jobs.