from glob import glob
import os
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms as T
from torchcodec.decoders import VideoDecoder
import numpy as np

class FrameImageDataset(torch.utils.data.Dataset):
    def __init__(self, 
    root_dir='/work3/ppar/data/ucf101',
    split='train', 
    transform=None
):
        self.frame_paths = sorted(glob(f'{root_dir}/frames/{split}/*/*/*.jpg'))
        self.df = pd.read_csv(f'{root_dir}/metadata/{split}.csv')
        self.split = split
        self.transform = transform
       
    def __len__(self):
        return len(self.frame_paths)

    def _get_meta(self, attr, value):
        return self.df.loc[self.df[attr] == value]

    def __getitem__(self, idx):
        frame_path = self.frame_paths[idx]
        video_name = frame_path.split('/')[-2]
        video_meta = self._get_meta('video_name', video_name)
        label = video_meta['label'].item()
        
        frame = Image.open(frame_path).convert("RGB")

        if self.transform:
            frame = self.transform(frame)
        else:
            frame = T.ToTensor()(frame)

        return frame, label


class FrameFlowImageDataset(torch.utils.data.Dataset):
    def __init__(self, 
    root_dir='/work3/ppar/data/ucf101',
    split='train', 
    transform=None
):
        self.frame_paths = sorted(glob(f'{root_dir}/frames/{split}/*/*/*.jpg'))
        self.df = pd.read_csv(f'{root_dir}/metadata/{split}.csv')
        self.split = split
        self.transform = transform
       
    def __len__(self):
        return len(self.frame_paths)

    def _get_meta(self, attr, value):
        return self.df.loc[self.df[attr] == value]

    def __getitem__(self, idx):
        frame_path = self.frame_paths[idx]
        flow_path = frame_path.replace('frames', 'flows').split('/')[:-1]
        flow_path = '/'.join(flow_path)
        video_name = frame_path.split('/')[-2]
        video_meta = self._get_meta('video_name', video_name)
        label = video_meta['label'].item()
        frame = Image.open(frame_path).convert("RGB")
        flows = self.load_flow_frames(flow_path)
        if self.transform:
            frame = self.transform(frame)
        else:
            frame = T.ToTensor()(frame)
        flows = [T.ToTensor()(flow) for flow in flows]
        flows = torch.stack(flows).permute(0, 2, 1, 3)

        return frame, flows, label
    
    def load_flow_frames(self, flow_path):
        flows = []
        for i in range(1, 10):
            flow_file = os.path.join(flow_path, f"flow_{i}_{i+1}.npy")
            flow = np.load(flow_file)
            flows.append(flow)
        return flows



class FrameVideoDataset(torch.utils.data.Dataset):
    def __init__(self, 
    root_dir = '/work3/ppar/data/ucf101', 
    split = 'train', 
    transform = None,
    stack_frames = True
):

        self.video_paths = sorted(glob(f'{root_dir}/videos/{split}/*/*.avi'))
        self.df = pd.read_csv(f'{root_dir}/metadata/{split}.csv')
        self.split = split
        self.transform = transform
        self.stack_frames = stack_frames
        
        self.n_sampled_frames = 10

    def __len__(self):
        return len(self.video_paths)
    
    def _get_meta(self, attr, value):
        return self.df.loc[self.df[attr] == value]

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        video_name = video_path.split('/')[-1].split('.avi')[0]
        video_meta = self._get_meta('video_name', video_name)
        label = video_meta['label'].item()

        video_frames_dir = self.video_paths[idx].split('.avi')[0].replace('videos', 'frames')
        video_frames = self.load_frames(video_frames_dir)

        if self.transform:
            frames = [self.transform(frame) for frame in video_frames]
        else:
            frames = [T.ToTensor()(frame) for frame in video_frames]
        
        if self.stack_frames:
            frames = torch.stack(frames).permute(1, 0, 2, 3)


        return frames, label
    
    def load_frames(self, frames_dir):
        frames = []
        for i in range(1, self.n_sampled_frames + 1):
            frame_file = os.path.join(frames_dir, f"frame_{i}.jpg")
            frame = Image.open(frame_file).convert("RGB")
            frames.append(frame)

        return frames


class VideoFileDataset(torch.utils.data.Dataset):
    def __init__(self,
        root_dir='/work3/ppar/data/ucf101',
        split='train',
        transform=None,
        num_frames=10,
        stack_frames=True
    ):
        self.video_paths = sorted(glob(f'{root_dir}/videos/{split}/*/*.avi'))
        self.df = pd.read_csv(f'{root_dir}/metadata/{split}.csv')
        self.split = split
        self.transform = transform
        self.num_frames = num_frames
        self.stack_frames = stack_frames

    def __len__(self):
        return len(self.video_paths)

    def _get_meta(self, attr, value):
        return self.df.loc[self.df[attr] == value]

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        video_name = video_path.split('/')[-1].split('.avi')[0]
        video_meta = self._get_meta('video_name', video_name)
        label = video_meta['label'].item()

        # Load video using torchcodec
        decoder = VideoDecoder(video_path)
        total_frames = len(decoder)

        # Sample frames uniformly across the video
        frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)

        frames = []
        for frame_idx in frame_indices:
            # Get frame as tensor [C, H, W] in uint8
            frame = decoder[frame_idx]
            # Convert to PIL Image for transforms
            frame = frame.permute(1, 2, 0).cpu().numpy()  # [H, W, C]
            frame = Image.fromarray(frame)

            if self.transform:
                frame = self.transform(frame)
            else:
                frame = T.ToTensor()(frame)

            frames.append(frame)

        if self.stack_frames:
            frames = torch.stack(frames).permute(1, 0, 2, 3)  # [C, T, H, W]

        return frames, label

class FlowFrameVideoDataset(torch.utils.data.Dataset):
    def __init__(self, 
    root_dir = '/dtu/datasets1/02516/ucf101_noleakage', 
    split = 'train', 
    stack_frames = True
):

        self.video_paths = sorted(glob(f'{root_dir}/videos/{split}/*/*.avi'))
        self.df = pd.read_csv(f'{root_dir}/metadata/{split}.csv')
        self.split = split
        self.stack_frames = stack_frames
        
        self.n_sampled_frames = 1 # only 1 for dual stream

    def __len__(self):
        return len(self.video_paths)
    
    def _get_meta(self, attr, value):
        return self.df.loc[self.df[attr] == value]

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        video_name = video_path.split('/')[-1].split('.avi')[0]
        video_meta = self._get_meta('video_name', video_name)
        label = video_meta['label'].item()

        video_frames_dir = self.video_paths[idx].split('.avi')[0].replace('videos', 'frames')
        video_frames = self.load_frames(video_frames_dir)
        flow_frames_dir = self.video_paths[idx].split('.avi')[0].replace('videos', 'flows')
        flows = self.load_flow_frames(flow_frames_dir)

        frames = [T.ToTensor()(frame) for frame in video_frames]
        flows = [T.ToTensor()(flow) for flow in flows]

        if self.stack_frames:
            frames = torch.stack(frames).permute(1, 0, 2, 3)
            flows = torch.stack(flows).permute(2, 0, 1, 3)

        return frames, flows, label
    
    def load_frames(self, frames_dir):
        frames = []
        for i in range(1, self.n_sampled_frames + 1):
            frame_file = os.path.join(frames_dir, f"frame_{i}.jpg")
            frame = Image.open(frame_file).convert("RGB")
            frames.append(frame)

        return frames
    
    def load_flow_frames(self, flow_dir):
        flows = []
        for i in range(1, self.n_sampled_frames):
            flow_file = os.path.join(flow_dir, f"flow_{i}_{i+1}.npy")
            flow = np.load(flow_file)
            flows.append(flow)

        return flows

if __name__ == '__main__':
    from torch.utils.data import DataLoader

    root_dir = '/dtu/datasets1/02516/ucf101_noleakage'

    transform = T.Compose([T.Resize((64, 64)),T.ToTensor()])
    frameimage_dataset = FrameImageDataset(root_dir=root_dir, split='val', transform=transform)
    framevideostack_dataset = FrameVideoDataset(root_dir=root_dir, split='val', transform=transform, stack_frames = True)
    framevideolist_dataset = FrameVideoDataset(root_dir=root_dir, split='val', transform=transform, stack_frames = False)
    flowframevideo_dataset = FlowFrameVideoDataset(root_dir=root_dir, split='val', stack_frames = True)

    frameimage_loader = DataLoader(frameimage_dataset,  batch_size=8, shuffle=False)
    framevideostack_loader = DataLoader(framevideostack_dataset,  batch_size=8, shuffle=False)
    framevideolist_loader = DataLoader(framevideolist_dataset,  batch_size=8, shuffle=False)
    flowframevideo_loader = DataLoader(flowframevideo_dataset,  batch_size=8, shuffle=False)

    # for frames, labels in frameimage_loader:
    #     print(frames.shape, labels.shape) # [batch, channels, height, width]

    # for video_frames, labels in framevideolist_loader:
    #     print(45*'-')
    #     for frame in video_frames: # loop through number of frames
    #         print(frame.shape, labels.shape)# [batch, channels, height, width]

    # for video_frames, labels in framevideostack_loader:
    #     print(video_frames.shape, labels.shape) # [batch, channels, number of frames, height, width]

    # for video_frames, flow_frames, labels in flowframevideo_loader:
    #     print(video_frames.shape, flow_frames.shape, labels.shape) # [channels, number of frames, height, width]
    
    frameflowimage_dataset = FrameFlowImageDataset(root_dir=root_dir, split='val', transform=transform)
    frameflowimage_loader = DataLoader(frameflowimage_dataset,  batch_size=8, shuffle=False)

    for frame, flow, labels in frameflowimage_loader:
        print(frame.shape, flow.shape, labels.shape) # [batch, channels, height, width]