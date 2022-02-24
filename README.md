# Decoding-Neural-Activity-to-Reconstruct-and-Generate-Music-Stimuli
This project uses deep regression and generative models to reconstruct music from brain responses as participants passively listen to full length music.   

The project was inspired by Brain2Image (Kavasidis et al., 2017) model where generative models took in EEG data of participants looking at images as input and outputted reconstructions of the image stimuli. 

For more details: Kavasidis, I., Palazzo, S., Spampinato, C., Giordano, D., & Shah, M. (2017, October). Brain2image: Converting brain signals into images. In Proceedings of the 25th ACM international conference on Multimedia (pp. 1809-1817).

In our project we continue this type of 'Mind Reading' research but with EEG responses to music. We take two directions, both a generative model approach and a neural decoding/signal processing approach. The latter is of focus as we posit a signal recovery functionality from CNN based deep regressions. Instead of learning a normal distribution of features from each stimulus class, we use deep regression to map a 1 to 1 from input to desired output. This puts the hypothesis foward that EEG signals can be treated as noisy versions of the auditory stimulus and allow for the continous reconstruction of a full length song instead of just the mean identity. For every 1 second of EEG data we can reconstruct 1 second of music the participant was listening to at that time. 

![alt text](Figures/Model_input_output1.png?raw=true)

![alt text](Figures/stim_rec1.png?raw=true)
