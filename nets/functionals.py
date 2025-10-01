from lib.lorentz.manifold import CustomLorentz
import torch
from torch.functional import Tensor
import numpy as np
from hsssw.lib.hhsw import horo_hyper_sliced_wasserstein_lorentz
from scipy.linalg import sqrtm, inv 
import os
import pickle

def euler_align(raw_array):
    """
    Perform Euler alignment on input numpy array ([trials * channels * samples]) 
    and return the aligned array.
    """
    # Calculate mean covariance matrix
    cov_matrices = [np.cov(trial, rowvar=True) for trial in raw_array]
    #cov_matrices = np.cov(raw_array,axis=0, rowvar=True)
    mean_cov_matrix = np.mean(cov_matrices, axis=0)
    
    # Compute transformation matrix
    trans_matrix = inv(sqrtm(mean_cov_matrix))
    
    # Apply transformation to all trials using broadcasting
    return trans_matrix @ raw_array


def swd_loss(x_flatten: Tensor, domains: Tensor):

    du = domains.unique()
    manifold = CustomLorentz()
    loss = 0
    for i in du:
        x = x_flatten[domains == i]
        x_gaussian = torch.normal(0, 1, size=x.shape, dtype=x.dtype, device=x.device)
        x_gaussian = x_gaussian / (x_gaussian.norm(p=2, dim=-1, keepdim=True) + 1e-8)
        x_gaussian = manifold.expmap0(x_gaussian)
        loss += horo_hyper_sliced_wasserstein_lorentz(x, x_gaussian, 1000, device=x.device, p=2)
    return loss

def load_dataset(path_):

    label_9class = np.array([
        0,0,0,
        1,1,1,
        2,2,2,
        3,3,3,
        4,4,4,4,
        5,5,5,
        6,6,6,
        7,7,7,
        8,8,8
    ], dtype=int)

    for i in range(123):
        path = f'{path_}/sub{i:03d}.pkl'
        with open(path, 'rb') as f:
            data = pickle.load(f)
        path_new = f'./faced_data/subject_{i}/session_1/'
        if not os.path.exists(path_new):
            os.makedirs(path_new, exist_ok=True)
            np.save(os.path.join(path_new, 'eeg.npy'), data)
            np.save(os.path.join(path_new, 'label.npy'), label_9class)

    all_data = []
    all_labels = []
    all_subjects = []
    all_sessions = []
    base_path ='./faced_data'
    # Get all subject folders
    subject_folders = [f for f in os.listdir(base_path) if f.startswith('subject_')]
    subject_folders.sort()  # Ensure consistent ordering
    
    for subject_folder in subject_folders:
        subject_path = os.path.join(base_path, subject_folder)
        if not os.path.isdir(subject_path):
            continue
            
        # Extract subject number
        subject_id = int(subject_folder.split('_')[1])
        
        # Get all session folders for this subject
        session_folders = [f for f in os.listdir(subject_path) if f.startswith('session_')]
        session_folders.sort()  # Ensure consistent ordering
        
        for session_folder in session_folders:
            session_path = os.path.join(subject_path, session_folder)
            if not os.path.isdir(session_path):
                continue
                
            # Extract session number
            session_id = int(session_folder.split('_')[1])
            
            # Load EEG data and labels
            eeg_file = os.path.join(session_path, 'eeg.npy')
            label_file = os.path.join(session_path, 'label.npy')
            
            if os.path.exists(eeg_file) and os.path.exists(label_file):
                try:
                    eeg_data = np.load(eeg_file)
                    labels = np.load(label_file)
                    
                    # Ensure data shapes are compatible
                    if eeg_data.shape[0] != labels.shape[0]:
                        print(f"Warning: Shape mismatch in {subject_folder}/{session_folder}")
                        continue
                    
                    # Store data
                    all_data.append(eeg_data)
                    all_labels.append(labels)
                    
                    # Create subject and session arrays for this batch
                    n_trials = eeg_data.shape[0]
                    all_subjects.extend([subject_id] * n_trials)
                    all_sessions.extend([session_id] * n_trials)
                    
                    print(f"Loaded {subject_folder}/{session_folder}: {eeg_data.shape[0]} trials")
                    
                except Exception as e:
                    print(f"Error loading {subject_folder}/{session_folder}: {e}")
            else:
                print(f"Missing files in {subject_folder}/{session_folder}")
    
    if not all_data:
        raise ValueError("No data loaded. Check the path and file structure.")
    
    # Concatenate all data
    data = np.concatenate(all_data, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    subjects = np.array(all_subjects)
    sessions = np.array(all_sessions)
    
    return data, labels, subjects, sessions