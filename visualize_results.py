import matplotlib.pyplot as plt
import numpy as np
import torch

def _assemble_grid_elements(prediction_outputs):
    # Extract and concatenate results from all batches
    all_images = []
    all_masks = []
    all_predictions = []
    
    for batch_result in prediction_outputs:
        all_images.append(batch_result['images'])
        all_masks.append(batch_result['masks'])
        all_predictions.append(batch_result['predictions'])
    
    # Concatenate all batches
    images = torch.cat(all_images, dim=0)
    masks = torch.cat(all_masks, dim=0)
    predicted_masks = torch.cat(all_predictions, dim=0)
    return images, masks, predicted_masks


def visualization_grid(prediction_outputs):
    
    # Convert tensors to numpy arrays if needed
    images, masks, predicted_masks = _assemble_grid_elements(prediction_outputs)

    if torch.is_tensor(images):
        images = images.cpu().numpy()
    if torch.is_tensor(masks):
        masks = masks.cpu().numpy()
    if torch.is_tensor(predicted_masks):
        predicted_masks = predicted_masks.cpu().numpy()

    samples_to_visualize = len(images)
    
    # Create figure with subplots
    fig, axes = plt.subplots(samples_to_visualize, 3, figsize=(12, 4 * samples_to_visualize))
    
    # Handle case where there's only one sample
    if samples_to_visualize == 1:
        axes = axes.reshape(1, -1)
    
    # Column titles
    titles = ['Original Image', 'Ground Truth Mask', 'Predicted Mask']

    mean = np.array([0.5543, 0.3644, 0.2777])
    std = np.array([0.2840, 0.2101, 0.1770])
    
    for i in range(samples_to_visualize):
        # Original image
        if images[i].ndim == 3:
            # Handle RGB images (H, W, C) or (C, H, W)
            if images[i].shape[0] == 3:  # (C, H, W) format
                img_display = np.transpose(images[i], (1, 2, 0))
            else:  # (H, W, C) format
                img_display = images[i]

            # Denormalize the image
            img_display = img_display * std + mean
            img_display = np.clip(img_display, 0, 1)
        else:
            # Grayscale image
            img_display = images[i]
        
        axes[i, 0].imshow(img_display)
        axes[i, 0].set_title(titles[0] if i == 0 else '')
        axes[i, 0].axis('off')
        
        # Ground truth mask
        if masks[i].ndim == 3:
            mask_display = masks[i].squeeze()
        else:
            mask_display = masks[i]
        
        axes[i, 1].imshow(mask_display, cmap='gray')
        axes[i, 1].set_title(titles[1] if i == 0 else '')
        axes[i, 1].axis('off')
        
        # Predicted mask
        if predicted_masks[i].ndim == 3:
            pred_display = predicted_masks[i].squeeze()
        else:
            pred_display = predicted_masks[i]
        
        axes[i, 2].imshow(pred_display, cmap='gray')
        axes[i, 2].set_title(titles[2] if i == 0 else '')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    return fig

