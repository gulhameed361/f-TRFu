# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 16:03:19 2024

@author: Gul Hameed
"""

import numpy as np

def compute_eigenvalues_from_sparse(hessian_entries, var_list_length):
    """
    Compute the eigenvalues of a Hessian matrix given in sparse format 
    and extend it to match the full variable list length.
    
    Parameters:
    hessian_entries (list of lists): A list where each sublist is of the form [i, j, value],
                                     representing the Hessian matrix entry at row i and column j.
    var_list_length (int): Total number of variables in the model.

    Returns:
    tuple: 
        - hessian_matrix (numpy.ndarray): Full Hessian matrix with extended dimensions.
        - eigenvalues (numpy.ndarray): Eigenvalues of the Hessian matrix.
        - H_abs (numpy.ndarray): Reconstructed Hessian matrix with absolute eigenvalues.
        - H_clamped (numpy.ndarray): Reconstructed Hessian matrix with eigenvalues clamped to zero for negative values.
    """
    # Initialize a square Hessian matrix of size equal to the total number of variables
    hessian_matrix = np.zeros((var_list_length, var_list_length))
    
    # Populate the Hessian matrix using the sparse entries
    for i, j, value in hessian_entries:
        hessian_matrix[i, j] = value
        hessian_matrix[j, i] = value  # Ensure symmetry
    
    # Enforce symmetry explicitly to avoid floating-point errors
    hessian_matrix = (hessian_matrix + hessian_matrix.T) / 2
    
    # Compute the eigenvalues and eigenvectors of the Hessian matrix
    eigenvalues, eigenvectors = np.linalg.eigh(hessian_matrix)
    
    # Smaller absolute values are clamped for numerical stability
    # Replace eigenvalues with their absolute values
    abs_eigenvalues = np.maximum(np.abs(eigenvalues), 1e-8)
    
    # Reconstruct the Projected Hessian matrix with absolute eigenvalues
    H_abs = eigenvectors @ np.diag(abs_eigenvalues) @ eigenvectors.T
    
    # Numerical check: Ensure H_clamped is PSD
    assert np.all(np.linalg.eigvalsh(H_abs) >= 0), "H_abs is not PSD!"
    
    # Clamp negative eigenvalues to zero
    # Clamping to a small value rather than 0 to ensure numerical stability and avoiding singular matrix
    # However, zero is more accurate as it preserves more accuracy
    clamped_eigenvalues = np.maximum(eigenvalues, 1e-8) 
    
    # Reconstruct the Projected Hessian matrix with clamped eigenvalues
    H_clamped = eigenvectors @ np.diag(clamped_eigenvalues) @ eigenvectors.T
    
    # Numerical check: Ensure H_clamped is PSD
    assert np.all(np.linalg.eigvalsh(H_clamped) >= 0), "H_clamped is not PSD!"
    
    return hessian_matrix, eigenvalues, abs_eigenvalues, H_abs, clamped_eigenvalues, H_clamped


