# # SRGAN
import os
import matplotlib.pyplot as plt

from data import DIV2K, DATASET
from image_loader import data_loader
from model.srgan import generator, discriminator
from train import SrganTrainer, SrganGeneratorTrainer

# Location of model weights (needed for demo)
weights_dir = 'weights/srgan'
weights_file = lambda filename: os.path.join(weights_dir, filename)

SCALE = 1
DOWNGRADE = 'bicubic'

os.makedirs(weights_dir, exist_ok=True)

train = DATASET(scale=SCALE, downgrade=DOWNGRADE)
train_ds = train.build_dataset(batch_size=8, random_transform=True, min_jpeg_compression=30, max_jpeg_compression=75)

valid = DATASET(scale=SCALE, training='validation', downgrade=DOWNGRADE)
valid_ds = valid.build_dataset(batch_size=8, random_transform=True, min_jpeg_compression=50, max_jpeg_compression=85)

pre_trainer = SrganGeneratorTrainer(model=generator(), checkpoint_dir=f'.ckpt/pre_generator')
pre_trainer.train(train_ds,
                  valid_ds.take(10),
                  steps=6000,
                  evaluate_every=1000, 
                  save_best_only=False)

pre_trainer.model.save_weights(weights_file('pre_generator.h5'))

gan_generator = generator()
gan_generator.load_weights(weights_file('pre_generator.h5'))

gan_trainer = SrganTrainer(generator=gan_generator, discriminator=discriminator())
gan_trainer.train(div2k_train, steps=200000)

gan_trainer.generator.save_weights(weights_file('gan_generator.h5'))
gan_trainer.discriminator.save_weights(weights_file('gan_discriminator.h5'))

