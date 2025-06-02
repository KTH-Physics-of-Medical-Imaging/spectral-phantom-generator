#import SimpleITK as sitk
from totalsegmentator.python_api import totalsegmentator
from totalsegmentator.map_to_binary import class_map
from typing import Union
from nibabel import Nifti1Image
import nibabel as nib
from pathlib import Path 
import argparse, os
from tqdm import tqdm
from data.parameters import CT_FILENAME, ALL_SEGMENTATION_TASKS

class Segmenter:
    """
    Segmenter class for handling medical image segmentation tasks.
    This class provides an interface for segmenting medical images using various predefined tasks.
    It supports setting input images, specifying output directories, and saving segmentation results.
    Attributes
    ----------
    input : Union[str, Path, Nifti1Image], optional
        The input image to be segmented. Can be a file path, Path object, or a Nifti1Image.
    output_dir : str or Path, optional
        Directory where segmentation outputs will be saved.
    license_number : str, optional
        License number required for segmentation tools, if applicable.
    Methods
    -------
    set_output_dir(output_dir)
        Set the directory where segmentation outputs will be saved.
    set_input(input)
        Set the input image for segmentation.
    segment_multitasks(tasks=['total', 'tissue_4_types', 'body'], fast=False, save_segmentations=True)
        Perform segmentation for multiple tasks. Optionally saves the segmentation results.
    segment_subtask(task='total', fast=False)
        Perform segmentation for a single specified task.
    save_seg(seg, path)
        Save a segmentation result to the specified file path.
    """

    
    def __init__(self, output_dir=None, license_number=None):
        self.input = None
        self.output_dir = output_dir
        self.license_number = license_number

    def set_output_dir(self, output_dir):
        self.output_dir = output_dir

    def set_input(self, input: Union[str, Path, Nifti1Image]):
        self.input = input

    def segment_multitasks(self, tasks=['total', 'tissue_4_types', 'body'], fast=False, save_segmentations=True):
        assert self.input is not None, 'Input is not set'
        subsegmentations = []
        for task in tasks:
            subsegmentations.append(self.segment_subtask(task=task, fast=fast))
            if save_segmentations:
                self.save_seg(subsegmentations[-1], Path(self.output_dir, f'{task}.nii.gz'))

    def segment_subtask(self, task='total', fast=False): # total, tissue_4_types, body
        assert self.input is not None, 'Input is not set'
        seg = totalsegmentator(self.input, task=task, fast=fast, license_number=self.license_number)
        return seg
    
    def save_seg(self, seg, path):
        print(f'Saving segmentation to {path}')
        nib.save(seg, path)
    



if __name__ == '__main__':
    import json
    from data.parameters import ALL_SEGMENTATION_TASKS
    
    parser = argparse.ArgumentParser(description='Segment organs and body from CT images using TotalSegmentator')
    parser.add_argument('--dir', type=str, help='The main directory of the dataset', default='/Users/joelva/work/kits23')
    parser.add_argument('--ct_filename', type=str, help='The filename of the input ct file', default=CT_FILENAME)
    parser.add_argument('--cases', type=str, default='0:1', help='Slice or list of cases to process, e.g. "0:10" or "1,2,3"')
    parser.add_argument('--license_number', type=str, help='License number for TotalSegmentator', default=None)
    parser.add_argument('--fast', action='store_true', default=False, help='Use fast mode (model trained on 3 mm slices) for segmentation. If false, the fine model trained on 1 mm slices will be used.')
    args = parser.parse_args()
    
    args.tasks = ALL_SEGMENTATION_TASKS

    if ':' in args.cases:
        start, end = args.cases.split(':')
        cases = slice(int(start) if start else None, int(end) if end else None)
    elif ',' in args.cases:
        cases = [int(x) for x in args.cases.split(',')]
    else:
        cases = slice(int(args.cases), int(args.cases)+1)


    subdirs = sorted([d for d in os.listdir(args.dir) if Path(args.dir, d).is_dir()]) # sort dirs
    subdirs = subdirs[cases] 
  

    print(f'Performing the following tasks: {args.tasks}')
    print(f'Processing {len(subdirs)} cases: {subdirs}')
  
    segmenter = Segmenter(license_number=args.license_number)

    for subdir in tqdm(subdirs, desc='Processing cases...'):
    
        subdir = Path(args.dir, subdir)

        segmenter.set_output_dir(subdir)
        segmenter.set_input(Path(subdir, args.ct_filename))
        segmenter.segment_multitasks(tasks=args.tasks, 
                                     save_segmentations=True, 
                                     fast=args.fast)
        
    print('Finished segmenting all cases.')