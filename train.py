import torch 
import gymnasium as gym
import argparse


from models.GMM import MLP
from tqdm import tqdm
from utils.dataset import TrajectoryDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter


def distill(name):

    writer = SummaryWriter(f"./runs/{name}")

    # Load data
    dataset = TrajectoryDataset(data_dir="./data/ds-2024-08-13_15-40-47")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize model info = dataset.get_info()
    info = dataset.get_info()
    action_space = info["action_space"]
    action_space = gym.spaces.flatten_space(action_space).shape[0]
    obs_space = gym.spaces.flatten_space(info["obs_space"]).shape[0]
    # Initialize model
    model = MLP(
        input_dim=obs_space,
        output_dim=action_space
    ).cuda()

    optim = torch.optim.Adam(model.parameters(), lr=0.001)

    # # working on collecting reaching data to train a simple MLP model, 
    # # eventually want GMM
    for i in tqdm(range(200)):
        running_loss = 0
        for step, (obs, act) in enumerate(dataloader):
            optim.zero_grad()
            obs = obs.cuda()
            act = act.cuda()

            pred = model(obs)

            loss = torch.nn.functional.mse_loss(pred, act)
            loss.backward()
            optim.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        writer.add_scalar("Loss/train", avg_loss, i)

        ckpt_path = f"./runs/{name}/model_{i}.pth"
        if i % 50 == 0:
            writer.add_scalar("checkpoints", i, i)
            torch.save({
                "epoch": i,
                "loss": avg_loss,
                "model_state_dict": model.state_dict(),
                "optim_state_dict": optim.state_dict()
                }, ckpt_path)

    writer.flush()
    writer.close()





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="training!")
    parser.add_argument("--name", type=str, required=True, help="Name of this experiment.")
    args_cli = parser.parse_args()

    distill(args_cli.name)

