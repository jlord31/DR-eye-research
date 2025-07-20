import torch
import numpy as np

def extract_features(loader, model, device):
    """
        Extract features from a dataset using a given model.
        Args:
            loader (DataLoader): DataLoader for the dataset
            model (nn.Module): Model to extract features from
            device (torch.device): Device to run the model on
        Returns:
            features (np.ndarray): Extracted features
            labels (np.ndarray): Corresponding labels
            paths (list): List of file paths (if available)
    """
    
    model.eval()
    feat_list = []
    label_list = []
    path_list = []

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 2:
                imgs, lbls = batch
                paths = None
            elif len(batch) == 3:
                imgs, lbls, paths = batch
            else:
                raise ValueError(f"Unexpected batch format with {len(batch)} elements")

            imgs = imgs.to(device)

            x = model.patch_embed(imgs)
            batch_size = imgs.shape[0]
            cls_tokens = model.cls_token.expand(batch_size, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            x = model.pos_drop(x + model.pos_embed)
            for blk in model.blocks:
                x = blk(x)
            x = model.norm(x)
            cls_features = x[:, 0]

            feat_list.append(cls_features.cpu().numpy())
            label_list.extend(lbls.cpu().numpy())

            if paths is not None:
                path_list.extend(paths)

    features = np.vstack(feat_list)
    labels = np.array(label_list)

    if path_list:
        return features, labels, path_list
    else:
        return features, labels
