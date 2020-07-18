## Bert Classifier
  My first work using Bert in pure pytorch. The dataset used is the IMDB movie reviews dataset. Total Trianing time on GPU was 1hr 26mins for as single epoch. An accuracy of 93.95% was gotten on the validation set
  
  ### How To:
    The way this is structured is:
      - Obtaing the BertModel and the BertTokenizer from Huggingface. Note that a normal tokenizer from the Tokenizers library can also be used.
      - Build the BertModel. Note that the body of the model from huggingface outputs logits of size 768 so a linear classifier head shoud be placed on top of this body with the right number of classes that we need
      - Create a Dataset class soo we can make a Dataloader. Make sure to handles padding just as i have shown
      - Every other thing is pretty staright forward and can be understood by any pytorch dev
      - Flas endpoints have also been created
      
      
## XLM_ROberta_on_TPUs
  This is an implementation of a multilanguage classification problem using multilingual XLM-Roberta. I used a 5 fold cross validation strategy and trained each for 10 epochs.The training was done on TPU and the the total training time was ~40 hours across all folds. An Accuracy of about ~96% was obtained across all the folds.
  
  ### How To:
    I order to move training to GPU, one must do the following:
      - Install torch_xla and all it's dependencies
      - Create a muliprocessor model wrapper `xmp.MpModelWrapper(__your_model__)` for the model to ensure that they are only loaded once into memory by a single process. NB: Do not forget to send your model `to(device)` after wrapping it.
      - Change the device(obviously) to `xm.xla_device`

      - Add a `DistributedSampler` as a sample to both the training and validation dataloaders. This is because we want the data to be gotten by multiple processes running in parallel
          - The number of replicas in this sampler. This tells us the number of processes that are participating in the distributed training. This can be gotten from `xm.xrt_world_size`
          - The rank: tels us the rank or the ordering of the processes in memory that have been assigned to handle the training job. It can be gotten from `xm.get_ordinal()`

      - Ensure to put dataloaders (train and valid) into `pl.ParallelLoader(loader, device)`...this ensures that they can be multiporcessed properly

      - In the Dataloaders (train and valid) ensure you use `drop_last=True` to avoid TPU errors

      - Ensure that the learning rate is spread across all the processes by multiplying it by `xm_xrt_world_size()`

      #### During training
      - Ensure to wrap the entire training oop into a function. This is important as we will need to spawn it later to pass to the different processors

      - We can optionally define a tracker `xm.RateTracker()` to see the state of the TPU dusing each minibatch step.
          - set `tracker = xm.RateTracker()`
          - add the batchsize to the tracker by doing `tracker.add(__your_bs__)`
          - Report the processor order 
          - Optionally report the tracker rate `tracker.rate()`, global rate `tracker.global_rate()`, the time taken `time.asctime()`
          - flush the text stream being printed

      - While training, wrap the desired optimizer for our problem inside `xm.optimizer_step()`

      - Always use `xm.master_print()` for printing

      ######Start training##########
      - Change the default tensor type to floattensor by doing `torch.set_default_tensor_type('torch.FloatTensor)`

      wrap the training loop func inside another func and pass it into `xmp.spawn(wrapper_func, nprocs=num_processoer(usually 8), start_method='fork')`

      NB: It is important to note that everything should be wrapped around the world func whihc is then wrapped around `wrapper_func`
      
 Everything else if quite straight-forward
