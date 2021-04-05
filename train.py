import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from model import Discriminator, Generator, initialize_weights
from torchvision.transforms import transforms
from PIL import Image
import os

#hyperparams
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 2e-4
BATCH_SIZE = 1
IMAGE_SIZE = 64
CHANNELS_IMG = 1
Z_DIM = 100
NUM_EPOCHS = 22
FEATURES_DISC = 64
FEATURES_GEN = 64

transforms = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Resize(IMAGE_SIZE),
    ]
)


class CustomDataSet(Dataset):
    def __init__(self, dir, transform):
        self.dir = dir
        self.transform = transform
        self.ls = os.listdir(dir)

    def __len__(self):
        return len(self.ls)

    def __getitem__(self, item):
        img_path = os.path.join(self.dir, self.ls[item])
        image = Image.open(img_path)
        image = self.transform(image)
        return image, 0


dataset = CustomDataSet('./data', transforms)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
disc = Discriminator(CHANNELS_IMG, FEATURES_DISC).to(device)
initialize_weights(gen)
initialize_weights(disc)

opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

criterion = nn.BCELoss()

fixed_noise = torch.randn(8, Z_DIM, 1, 1).to(device)
writer_real = SummaryWriter(f'logs/real')
writer_fake = SummaryWriter(f'logs/fake')
step = 0



if __name__ == '__main__':
    gen.train()
    disc.train()

    for epoch in range(NUM_EPOCHS):
        for batch_idx, (real, _) in enumerate(loader):
            real = real.to(device)
            noise = torch.randn((BATCH_SIZE, Z_DIM, 1, 1)).to(device)
            fake = gen(noise)
            # TRAIN DISC max log(D(X)) + log(1-D(G(Z)))
            disc_real = disc(real).reshape(-1)
            loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
            disc_fake = disc(fake.detach()).reshape(-1)
            loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            loss_disc = (loss_disc_fake + loss_disc_real) / 2
            disc.zero_grad()
            loss_disc.backward(retain_graph=True)
            opt_disc.step()

            # TRAIN GENERATOR min log(1-D(G(Z))) or max log(D(G(Z))
            output = disc(fake).reshape(-1)
            loss_gen = criterion(output, torch.ones_like(output))
            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            if batch_idx % 3 == 0:
                print(
                    f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \
                                  Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}"
                )

                with torch.no_grad():
                    fake = gen(fixed_noise)
                    img_grid_real = torchvision.utils.make_grid(
                        real[:8], normalize=True
                    )
                    img_grid_fake = torchvision.utils.make_grid(
                        fake[:8], normalize=True
                    )

                    writer_real.add_image("Real", img_grid_real, global_step=step)
                    writer_fake.add_image("Fake", img_grid_fake, global_step=step)

                step += 1

    torch.save(gen.state_dict(), './model/gen_state_dict.pt')
    torch.save(disc.state_dict(), './model/dist_state_dict.pt')