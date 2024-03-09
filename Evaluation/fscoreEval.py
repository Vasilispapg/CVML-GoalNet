import numpy as np
import h5py
import json
import os

def decode_titles(encoded_titles, hdf5_file):
    decoded_titles = []
    for ref_array in encoded_titles:
        # Handle the case where each ref_array might contain multiple references
        for ref in ref_array:
            # Dereference each HDF5 object reference to get the actual data
            title_data = hdf5_file[ref]
            # Decode the title
            decoded_title = ''.join(chr(char[0]) for char in title_data)
            decoded_titles.append(decoded_title)
    return decoded_titles


def load_mat_file(file_path,videoID):
    """
    Load a .mat file and return its contents.

    :param file_path: Path to the .mat file.
    :return: Contents of the .mat file.
    """
    with h5py.File(file_path, 'r') as file:
        user_anno_refs=file['tvsum50']['user_anno'][:] # type: ignore
        video_refs=file['tvsum50']['video'][:] # type: ignore

        decoded_videos = decode_titles(video_refs,file)
    
        annotations = []        
        # Get the index from decoded video list to find the annotation for the video
        index = [i for i, x in enumerate(decoded_videos) if x.lower() in videoID.lower()][0]
        
        # Iterate over each reference
        for ref in user_anno_refs:
            # Dereference each HDF5 object reference
            ref_data = file[ref[0]]

            # Convert to NumPy array and add to the annotations list
            annotations.append(np.array(ref_data))
            
        return annotations[index]
    
    
def evaluate_summary(predicted_summary, user_summary, eval_method='avg'):
    max_len = max(len(predicted_summary), user_summary.shape[1])
    S = np.zeros(max_len, dtype=int)
    G = np.zeros(max_len, dtype=int)
    S[:len(predicted_summary)] = predicted_summary

    f_scores = []
    for user in range(user_summary.shape[0]):
        G[:user_summary.shape[1]] = user_summary[user]
        overlapped = S & G
        
        precision = sum(overlapped) / sum(S) if sum(S) != 0 else 0
        recall = sum(overlapped) / sum(G) if sum(G) != 0 else 0
        f_score = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0
        f_scores.append(f_score * 100)  # multiplied by 100 for percentage

    if eval_method == 'max':
        return max(f_scores)
    else:  # 'avg'
        return sum(f_scores) / len(f_scores)


def saveResults(videoID,f_score_max,f_score_avg):
    # save the results in a file
    results={}
    if(not os.path.exists('results.json')):
        with open('results.json', 'w') as file:
            json.dump(results, file)
    with open('results.json','r') as file:
        results=json.load(file)
        
    results[videoID]={'f_score_max':f_score_max,'f_score_avg':f_score_avg}
    with open('results.json', 'w') as file:
        json.dump(results, file)
    
def evaluation_method(ground_truth_path,summary_indices,videoID):
    
    # Get the ground_truth
    ground_truth = np.array(load_mat_file(ground_truth_path, videoID))

    f_score_max = evaluate_summary(summary_indices, ground_truth, 'max')
    f_score_avg = evaluate_summary(summary_indices, ground_truth)
    
    saveResults(videoID,f_score_max,f_score_avg)
    
    print(f'F-scoreA: {f_score_avg:.2}%')
    print(f'F-scoreM: {f_score_max:.2}%')
    