# Visual Explanations for Object Detection in Transformer-based architectures using Relevance Propagation

- **Still under development**: Repository for my thesis project. It focuses on enhancing interpretability in transformer-based architectures 
for object detection tasks.
- This project is built upon prior research outlined in [Generic Attention-model Explainability for Interpreting Bi-Modal and Encoder-Decoder Transformers](https://openaccess.thecvf.com/content/ICCV2021/papers/Chefer_Generic_Attention-Model_Explainability_for_Interpreting_Bi-Modal_and_Encoder-Decoder_Transformers_ICCV_2021_paper.pdf).
- Currently working with the [DETR model](https://huggingface.co/docs/transformers/main/en/model_doc/detr) from Hugging Face's Transformers library.
- The algorithm generates a relevance map of the image tokens wrt to a query token (detection).

# Results

- Visualization are presented in `Tensorboard`. 
- Here we can see the relevance maps for the two detections of cats in the image:

<p align="center">
  <img src="./resources/results/cat/cat_left.png" width="39%" />
  <img src="./resources/results/cat/cat_right.png" width="39%" /> 
</p>

- We can also see how tokens interact with each other through the attention maps.
- These are the attention maps for the left cat in the last encoder block for the most relevant image token (4, 13):

![Left Cat Encoder Attentions](./resources/results/cat/cat_left_encoder_attentions.png)

- And these are the attention maps for the left cat in the last decoder block for the query token:

![Left Cat Decoder Attentions](./resources/results/cat/cat_left_decoder_attentions.png)
