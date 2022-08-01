# imports: 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Callable, Union, Any, TypeVar, Tuple
# from torch import tensor as Tensor
# import abstractmethod: 
from abc import abstractmethod
from sklearn.manifold import TSNE
import torchvision.transforms as transforms
from torch.utils.data import Dataset

Tensor = TypeVar('torch.tensor')



# ---


class config: 
    image_path = "c:/temp/dataset_final/islands/images/ISL64xSeqRGBTrain5/"




class BaseVAE(nn.Module):

    
    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size:int, current_device: int, **kwargs) -> Tensor:
        raise NotImplementedError

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass

# %%
class Encoder(nn.Module):
    def __init__(self,
        in_channels: int,
        latent_dim: int,
        hidden_dims = None,  
        **kwargs) -> None:
        super(Encoder, self).__init__()

        
        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                                kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 4, latent_dim)

    def forward(self, inp):
        result = self.encoder(inp)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

# %%
class VanillaVAE(BaseVAE):


    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(VanillaVAE, self).__init__()

        self.latent_dim = latent_dim
        self.encoder = Encoder(in_channels, latent_dim, hidden_dims)

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())

    def encode(self, input: Tensor) -> List[Tensor]:
        return self.encoder(input)


    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, input)


        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]

    def save_model(self, step):
        torch.save(self.state_dict(), './ghostnet_' + str(step) + '.pt')

class CustomDataset(Dataset):

    def __init__(self, root, transform):
        # if not isinstance(root, tuple): raise # FIX THIS WHY TUPLE....
        # root = root[0]
        self.root = root
        self.transforms = transform
        self.ids = [os.path.split(i)[1].split('.jpg')[0] for i in glob.glob(root + '/*.jpg', recursive=True)]
        self.labelpath = Path(root).parent.parent / "labels" / Path(root).name

    def __getitem__(self, index):
        img_id = self.ids[index]

        imgname = img_id + '.jpg'

        with open(self.labelpath / (img_id + '.txt')) as f:
            target = f.readline()[0]
        assert (len(target)==1)
        target=int(target)
        #TODO Check if convert RGB makes sense for Grayscale
        img = Image.open(os.path.join(self.root, imgname)).convert('RGB')
        img = self.transforms(img)
        return img, target


    def __len__(self):
        return len(self.ids)


model = VanillaVAE(3, 128)
model.load_state_dict(torch.load(config.ptpath))

tsne_loader = CustomDataset('C:/temp/data_final/islands/images/ISL64xSeqRGBVal20/', transforms.ToTensor())

labels = []
results = []
for i in range(tsne_loader.__len__()):
    inp, label = tsne_loader[i]
    inp = inp[None, :]
    res = model.encode(inp.to('cuda'))[0].squeeze()
    results.append(res.cpu().detach().numpy())
    labels.append(label)

tsne_results = TSNE(n_components=2).fit_transform(results)

# https://www.kaggle.com/code/code1110/are-there-clusters-pca-tsne-vae/notebook
from matplotlib.cm import get_cmap
cmap = get_cmap("tab10")

labels = np.array(labels)


fig, ax = plt.subplots(1, 1, figsize=(12, 8))

for i in range(5):
    marker = "$" + str(i) + "$"
    idx = labels == i
    ax.scatter(tsne_results[idx, 0], tsne_results[idx, 1],
              marker=marker, color=cmap(i))
ax.set_title("t-SNE")
