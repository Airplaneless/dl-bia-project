import random
import torch
import matplotlib.pyplot as plt
import numpy as np


def show_kspace(kspace_list):
    """
    Plot a list of kspaces of one kspace
    """
    n = len(kspace_list)
    
    fig, ax = plt.subplots(1,n,figsize=(30, 4 * 5),
                           subplot_kw=dict(frameon=False, xticks=[], yticks=[]),
                           gridspec_kw=dict(wspace=0.0, hspace=0.0))
    if n == 1:
        ax.imshow((kspace_list[0][0].abs() + 1e-11).log())
    else:
        for i in range(n):
            ax[i].imshow((kspace_list[i][0].abs() + 1e-11).log()) 


def sum_to_one(n):
    values = [0.0, 1.0] + [random.random() for _ in range(n - 1)]
    values.sort()
    
    def toFixed(numObj, digits=3):
        return float(f"{numObj:.{digits}f}")

    per_cent_list = [toFixed(values[i+1] - values[i]) for i in range(n)]
    return per_cent_list


def calc_each_slice_contribution(sample_kspace, num_of_slices_per_artifact=4):
    """
    Returns a dictionary with number of slice and randomly sampled
    amount of this slice contribution. Slice numeration starts from 0
    """
    total_row_num = sample_kspace.shape[-1]
    
    percent_of_each_slice = sum_to_one(num_of_slices_per_artifact)
    rows_of_each_slice = [int(total_row_num * p) for p in percent_of_each_slice]
    
    if sum(rows_of_each_slice) != total_row_num:
        rows_residue = total_row_num - sum(rows_of_each_slice[:-1])
        rows_of_each_slice = rows_of_each_slice[:-1] 
        rows_of_each_slice.append(rows_residue)
    
    return rows_of_each_slice


def apply_mask_to_kspace(kspace_list, mask, dim):
    masked_kspace_list = []
    for idx, kspace in enumerate(kspace_list):
        aligned_mask = mask == idx
        masked_kspace_list.append(kspace[dim] * aligned_mask)
    final_kspace = torch.sum(torch.stack(masked_kspace_list), dim=0)
    return final_kspace
    
    
def add_motion_artefacts(dataset, pixels_of_each_slice, motion_artefact_coefficient=0.99, 
              num_of_slices_per_artifact=4): 

#     direction_list = ['left', 'right', 'top', 'bottom']
    direction_list = ['top', 'bottom']
    direction = random.choice(direction_list)
    
    if direction == 'left' or direction == 'right':  # TODO :right or left
        pass
    
    elif direction == 'bottom' or direction == 'top':
        prob = np.random.randint(100)
        if prob <= motion_artefact_coefficient * 100:

            kspace_list = [dataset[i][0] for i in range(num_of_slices_per_artifact)]
            motion_mask = torch.zeros([kspace_list[0].shape[-1], kspace_list[0].shape[-2]])

            # Sample Mask
            current_row = 0
            for idx, pixel_num in enumerate(pixels_of_each_slice[:-1]):
                motion_mask[current_row:pixel_num + current_row] = idx + 1
                current_row = current_row + pixel_num
            t_motion_mask = np.array([[motion_mask[j][i] for j in range(len(motion_mask))]
                                    for i in range(len(motion_mask[0]))])

            kspace = torch.stack((apply_mask_to_kspace(kspace_list, t_motion_mask, 0),
                                  apply_mask_to_kspace(kspace_list, t_motion_mask, 1)))
            
#             show_kspace([kspace])
            
# HOW TO CALL FUNCTIONS                                
# dataset = FastMRIh5Dataset(dir_train, FastMRITransform(RandomMaskFunc([0.08], [1])))
# pixels_of_each_slice = calc_each_slice_contribution(dataset[1][0], num_of_slices_per_artifact = 5)
# add_motion_artefacts(dataset, pixels_of_each_slice, num_of_slices_per_artifact = 5)