# CelebAMask-HQ++ and SemanticStyleGAN inversion manipulation
<sup>This is the main repository for the IBB project on FRI. The repository was created for a project that was part of Image Based Biometry course at the University of Ljubljana, Faculty for computer and information science.</sub>

To use this repository as intended you should have a NVIDIA GPU with appropriate NVIDIA driver and CUDA versions that are compatible with PyTorch and Tensorflow.

More instructions on how to use the repository can be found in the main branch of the repository.

## Abstract

In this work we present the extended version of the already existing CelebAMask-HQ dataset with images and corresponding segmentation masks, that allow for even more fine-grained control of the structure and texture of the facial region more specifically glasses. The extended version of the dataset, called CelebAMask-HQ++, adds manually annotated semantic masks of glasses lenses, glasses types and glasses landmarks.
In total 1548 images of people with glasses have been updated with segmentation mask, where the previous ‘eyeglasses’ has now been extended to ‘glasses frames’ and ‘glasses lenses’. Aditionally all the images of glasses were annotated with glasses landmarks and glasses types.
Finally, we explored and found better optimization schemes for embedding in SemanticStyleGAN latent space with the help of segmentation masks to get noticeably better segmentation masks and image embeddings, that yielded better results for downstream tasks like style transfer.

## New annotations

**Example of new landmark annotations, blended with segmentation masks and the original image**
![Image of a generated graph](/assets/overlayed_seg_maps.png)
Inversion results of an image, with regular optimization and with added segmentation mask, as well as the generator trained only with glasses images from the updated dataset.

**Example of new segmentation maps**
![Image of a generated graph](/assets/lenses.png)

**Processed segmentation maps with erosion**
![Image of a generated graph](/assets/erosion2.png)


## Improved inversion

**Improved inversion of both image and segmentation mask**
![Image of a generated graph](/assets/inversion_comparision.png)

**Improved generators with the new dataset**
![Image of a generated graph](/assets/comparison_segmentation_and_generators.png)


## Improved style mixing

**Style mixing before**
![Image of a generated graph](/assets/style_mixing_before1.png)

**Style mixing after**
![Image of a generated graph](/assets/style_mixing1.png)
