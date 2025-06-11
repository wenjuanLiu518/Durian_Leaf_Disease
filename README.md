# Image augmentation by a vision-language foundation model for durian leaf disease recognition
This study employs a vision–language foundation model (FLUX) fine-tuned via Low-Rank Adaptation (LoRA) to generate realistic durian leaf disease images from paired textual descriptions. The objective is to enrich limited datasets with semantically faithful synthetic samples—capturing variations in lighting, weather, and background—thereby improving the accuracy of downstream disease classifiers. The basic pipeline of the proposed framework is as follows:
![image](https://github.com/user-attachments/assets/9b55c31b-d6df-4e5f-9d53-8f86bf6088d0)

# Experimental Results
<img width="1182" alt="image" src="https://github.com/user-attachments/assets/2a0330e7-74bb-472a-92f3-b10effa9eb9b" />

Quantitative evaluations (KID and CAS) of generated synthetic images.

<img width="989" alt="image" src="https://github.com/user-attachments/assets/82a6ce04-af73-4e98-a8e3-f21a8ed4ab75" />
t-SNE Visualization of (a)Flux; (b) SD3.5; (c) SDXL

![image](https://github.com/user-attachments/assets/ee5cc2de-f519-4493-ae39-b958d521f1ca)

CAMs of DurioFLUX images

![image](https://github.com/user-attachments/assets/ee01754c-d5f6-4a02-b7e6-7d890abaf312)
Test accuracy comparison

# Technologies Used

# Dataset

# Usage Steps

# Citation
If you use this code in your research, please cite:
@article{ author = {Wenjuan Liu}, title = {"Image augmentation by a vision-language foundation model for durian leaf disease recognition"}, journal = {The Visual Computer}, year = {2025}, note = {Submitted} }
