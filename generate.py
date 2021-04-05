from model import Generator
import sys
import torch
from torchvision.utils import save_image
save_dir = './generated_images'

gen = Generator(100, 1, 64)
gen.load_state_dict(torch.load("model/gen_state_dict.pt"))
gen.eval()


if __name__ == '__main__':
    args = sys.argv
    if len(args) > 1:
        random_seed = int(args[1])
        print(f"Random seed is {random_seed}")
        torch.manual_seed(random_seed)
    else:
        pass
    noise = torch.randn((1, 100, 1, 1))
    output = gen(noise)
    print(output.shape)
    img = output[0]
    save_image(img, f'{save_dir}/output{random_seed}.png')
