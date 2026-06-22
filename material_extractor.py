import SimpleITK as sitk
import numpy as np
from scipy.ndimage import gaussian_filter, label
from image_reader import ImageReader
from tissue import Tissue, Atom, Bone, Fat, Lung, TissueAtomicCompositionTable
from utils import *
from case import Case
from data.parameters import SEGMENTATION_TASKS, BODY_SEGMENTATION_TASKS, CT_FILENAME

class MaterialExtractor:
    def __init__(self, extraction_materials, skip_tissue_type, skip_organ, atomic_composition_table: TissueAtomicCompositionTable):
        self.extraction_materials = extraction_materials
        self.skip_tissue_type = skip_tissue_type
        self.skip_organ = skip_organ
        self.atomic_composition_table = atomic_composition_table

    def set_case(self, case: Case):
        self.case = case
        self.segmentation = case.segmentation

    def apply_noise_to_attenuation(self, attenuation, shape=None, relative_sigma=0.01, smooth_sigma=.1):
        """Apply a small additive perturbation to attenuation values."""
        attenuation = np.asarray(attenuation)
        if attenuation.ndim == 0:
            if attenuation <= 0:
                return float(attenuation)
            noise = np.random.normal(0.0, relative_sigma * float(attenuation))
            return max(0.0, float(attenuation) + noise)

        noise_shape = attenuation.shape if shape is None else shape
        noise_scale = relative_sigma * np.abs(attenuation)
        noise = np.random.normal(0.0, noise_scale, noise_shape)
        if smooth_sigma is not None and smooth_sigma > 0:
            noise = gaussian_filter(noise, smooth_sigma)
        return np.clip(attenuation + noise, 0.0, None)

    def _dilation_kernel(self, radius):
        kernel = [radius] * 3
        if not self.case.continuous_slices:
            kernel[ImageReader.get_z_index(self.case.CT)] = 0
        return kernel

    def _component_structure(self, image):
        structure = np.ones((3, 3, 3), dtype=np.uint8)
        if not self.case.continuous_slices:
            z_index = ImageReader.get_z_index(image)
            structure = np.zeros((3, 3, 3), dtype=np.uint8)
            slices = [slice(None), slice(None), slice(None)]
            slices[z_index] = 1
            structure[tuple(slices)] = 1
        return structure

    def refine_tissue_masks_from_threshold(
        self,
        type_to_refine=Bone,
        hu_threshold=80,
        opening_radius=1,
        closing_radius=1,
        distance_smoothing_sigma=0.0,
    ):
        """Refine tissue labels by keeping thresholded CT components that intersect them."""
        segmentation = sitk.GetArrayFromImage(self.segmentation)
        refined_segmentation = segmentation.copy()

        threshold_mask = sitk.GetArrayFromImage(to_HU(self.case.CT, self.case.effective_kev)) > hu_threshold
        components, n_components = label(threshold_mask, structure=self._component_structure(threshold_mask))

        component_label = np.zeros(n_components + 1, dtype=segmentation.dtype)
        component_overlap = np.zeros(n_components + 1, dtype=np.int64)
        tissue_masks = {}
        refined_count = 0

        for organ_number, organ_name in self.case.segmentation_names.items():
            tissue = Tissue(organ_name, self.atomic_composition_table)
            if not isinstance(tissue.type, type_to_refine):
                continue

            old_mask = segmentation == organ_number
            tissue_masks[organ_number] = old_mask
            organ_components = components[old_mask]
            component_ids, overlaps = np.unique(organ_components[organ_components > 0], return_counts=True)
            for component_id, overlap in zip(component_ids, overlaps):
                if overlap > component_overlap[component_id]:
                    component_label[component_id] = organ_number
                    component_overlap[component_id] = overlap
            refined_count += 1

        threshold_labels = component_label[components]
        refined_labels = np.zeros_like(threshold_labels)
        for organ_number, old_mask in tissue_masks.items():
            organ_mask = sitk.GetImageFromArray(old_mask.astype(np.uint8))
            organ_mask.CopyInformation(self.segmentation)
            if opening_radius > 0:
                organ_mask = sitk.BinaryMorphologicalOpening(organ_mask, self._dilation_kernel(opening_radius))
            if closing_radius > 0:
                organ_mask = sitk.BinaryMorphologicalClosing(organ_mask, self._dilation_kernel(closing_radius))
            if distance_smoothing_sigma > 0:
                distance = sitk.SignedMaurerDistanceMap(
                    organ_mask,
                    insideIsPositive=True,
                    squaredDistance=False,
                    useImageSpacing=False,
                )
                distance = sitk.DiscreteGaussian(distance, variance=distance_smoothing_sigma**2)
                organ_mask = distance > 0

            smoothed_mask = sitk.GetArrayFromImage(organ_mask) > 0
            threshold_additions = (threshold_labels == organ_number) & smoothed_mask & ~old_mask
            refined_labels[old_mask | threshold_additions] = organ_number
        threshold_labels = refined_labels

        new_voxels = (threshold_labels > 0) & (segmentation != threshold_labels)
        refined_segmentation[new_voxels] = threshold_labels[new_voxels]

        refined_image = sitk.GetImageFromArray(refined_segmentation.astype(segmentation.dtype))
        refined_image.CopyInformation(self.segmentation)
        self.segmentation = refined_image
        self.case.segmentation = refined_image

        refined_voxels = int(np.count_nonzero(new_voxels))
        kept_components = int(np.count_nonzero(component_label[1:]))
        print(
            f'Refined {refined_count} {type_to_refine.__name__} labels from CT threshold: '
            f'kept {kept_components}/{n_components} components and added {refined_voxels} voxels '
            f'(HU>{hu_threshold})'
        )


class SubtractionExtractor(MaterialExtractor):
    """
    Extracts material-specific maps from CT images by subtracting the attenuation contributions of specified materials or tissues.
    This extractor is designed to generate virtual non-contrast (VNC) images or other material-decomposed images by removing the contribution of selected atoms or tissues from the CT data, based on segmentation and atomic composition information.
    Parameters
    ----------
    extraction_materials : list[Atom] | Atom | Tissue
        The materials (atoms or tissues) to extract or subtract from the CT image.
    atomic_composition_table : dict
        Table mapping tissue/material names to their atomic composition.
    material_fractions : dict, optional
        Dictionary mapping organ/tissue names (lowercase) to arrays of fractions for each material. Fractions are normalized to sum to 1.
    skip_atom : list[Atom] | Atom, optional
        Atoms to skip during extraction.
    skip_tissue_type : type | tuple[type], optional
        Tissue types to skip during extraction.
    clip_values : tuple(float | None, float | None), optional
        Minimum and maximum values to clip the resulting material maps.
    filter_result : bool, default False
        Whether to apply post-processing filtering to the extracted material maps.
    dilated_tissue_type_to_remove : type, optional
        Tissue type (e.g., Bone) to dilate and remove from the material maps to avoid leakage.
    Attributes
    ----------
    n_materials : int
        Number of materials to extract.
    material_fractions : dict
        Normalized material fractions for each tissue.
    clip_values : tuple
        Clipping range for output values.
    filter_result : bool
        Whether to filter the resulting material maps.
    dilated_tissue_type_to_remove : type or None
        Tissue type to remove by dilation.
    Methods
    -------
    extract(image=None)
        Extracts the subtraction of the segmentations, returning material maps and the leftover image.
    remove_dilated_tissue_type(material, type_to_remove=Bone)
        Dilates and removes specified tissue type from the material map.
    filter_material_map(material)
        Applies filtering and masking to the material map.
    """


    
    def __init__(self, extraction_materials, atomic_composition_table, material_fractions={}, 
                 skip_atom=None, skip_tissue_type=None, skip_organ=None, clip_values=(None, None), 
                 filter_result=False, dilated_tissue_type_to_remove=None, skip_unlisted_organs=False,
                 dilated_tissue_radius=1, dilated_tissue_hu_threshold=None,
                 dilated_tissue_hu_search_radius=None, remove_high_hu_from_material=False,
                 refine_bone_masks=False, bone_refinement_hu_threshold=100,
                 bone_refinement_opening_radius=1, bone_refinement_closing_radius=1,
                 bone_refinement_distance_smoothing_sigma=0.0,
                 material_opening_radius=0):
        super().__init__(extraction_materials, skip_tissue_type, skip_organ, atomic_composition_table)
        #assert isinstance(self.extraction_material, Atom), "Subtraction atom must be of type Atom"

        if isinstance(self.extraction_materials, Tissue):
            self.skip_atom = [Atom(a) for a in self.extraction_materials.atoms]
        elif isinstance(self.extraction_materials, Atom):
            self.skip_atom = [self.extraction_materials.name]
        elif isinstance(self.extraction_materials, list):
            self.skip_atom = [a.name for a in self.extraction_materials]

        if isinstance(skip_atom, list):
            self.skip_atom.append(*skip_atom)
        elif isinstance(skip_atom, Atom):
            self.skip_atom.append(skip_atom)

        self.n_materials = len(self.extraction_materials)
        self.material_fractions = material_fractions
        self.material_fractions = {k.lower(): v / np.sum(v) for k, v in material_fractions.items()} # normalize to sum to 1 and lower case

        self.clip_values = clip_values

        self.filter_result = filter_result

        if not isinstance(dilated_tissue_type_to_remove, list):
            self.dilated_tissue_type_to_remove = [dilated_tissue_type_to_remove]
        else:
            self.dilated_tissue_type_to_remove = dilated_tissue_type_to_remove
        self.dilated_tissue_radius = dilated_tissue_radius
        self.dilated_tissue_hu_threshold = dilated_tissue_hu_threshold
        self.dilated_tissue_hu_search_radius = dilated_tissue_hu_search_radius
        self.remove_high_hu_from_material = remove_high_hu_from_material
        self.refine_bone_masks = refine_bone_masks
        self.bone_refinement_hu_threshold = bone_refinement_hu_threshold
        self.bone_refinement_opening_radius = bone_refinement_opening_radius
        self.bone_refinement_closing_radius = bone_refinement_closing_radius
        self.bone_refinement_distance_smoothing_sigma = bone_refinement_distance_smoothing_sigma
        self.material_opening_radius = material_opening_radius

        self.skip_unlisted_organs = skip_unlisted_organs

    def set_case(self, case: Case):
        super().set_case(case)
        if self.refine_bone_masks:
            self.refine_tissue_masks_from_threshold(
                type_to_refine=Bone,
                hu_threshold=self.bone_refinement_hu_threshold,
                opening_radius=self.bone_refinement_opening_radius,
                closing_radius=self.bone_refinement_closing_radius,
                distance_smoothing_sigma=self.bone_refinement_distance_smoothing_sigma,
            )

    def extract(self, image=None):
        """ Extract the subtraction of the segmentations """

        image = self.case.CT if image is None else image

        ct_array = sitk.GetArrayFromImage(image)
        segmentation = sitk.GetArrayFromImage(self.segmentation)
        materials = np.zeros((self.n_materials, *ct_array.shape), dtype=ct_array.dtype)
        
        for organ_number, organ_name in self.case.segmentation_names.items():

            tissue = Tissue(organ_name, self.atomic_composition_table)
            
            if organ_name.lower() in map(str.lower, self.skip_organ if self.skip_organ is not None else []):
                print(f'Found {organ_name} and will skip it')
                continue
             
            if not self._is_relevant_extraction_organ(organ_name, tissue):
                print(f'Found {tissue.type} in {organ_name} and will skip it')
                continue

            if organ_name.lower() in self.material_fractions.keys(): # check if current organ name is in given material fractions list
                material_weights = self.material_fractions[organ_name.lower()]
            elif self.skip_unlisted_organs:
                continue # if no weights are given, assume no material contribution
            else:
                material_weights = np.zeros(self.n_materials)
                material_weights[0] = 1 # only keep first material if no weights are given


            organ_local_density = segmentation == organ_number # could be updated to allow fractions
            attenuation = tissue.linear_att_coeff(self.case.effective_kev, skip_atom=self.skip_atom)

            #if tissue.is_lesion:
            attenuation = self.apply_noise_to_attenuation(attenuation)

            attenuation = attenuation * organ_local_density # construct attenuation map for the organ
            organ_material = ct_array[organ_local_density] - attenuation[organ_local_density] # subtract to get material contribution within the organ

            material_weights = np.array(material_weights)
            material_weights = material_weights / material_weights.sum()
            material_weights = material_weights.reshape(self.n_materials, *(1,)*len(organ_material.shape))    
            organ_material = np.stack([organ_material] * self.n_materials, axis=0)
            organ_material = material_weights * organ_material
            attenuation = attenuation * organ_local_density
            
            materials[:, organ_local_density] = organ_material.clip(*self.clip_values)

        materials_ = []
        leftovers = self.case.CT

        for m in range(materials.shape[0]):

            material = materials[m]
            material = sitk.GetImageFromArray(material)
            material.CopyInformation(self.case.CT)

            print(material.GetSize(), material.GetSpacing(), material.GetOrigin(), material.GetDirection())
            if self.filter_result:
                material = self.filter_material_map(material)

            if self.dilated_tissue_type_to_remove is not None:
                for tissue_type in self.dilated_tissue_type_to_remove:
                    print('Removing dilated tissue type', self.dilated_tissue_type_to_remove)
                    material = self.remove_dilated_tissue_type(material, tissue_type)

            if self.material_opening_radius > 0:
                material = self.open_material_map(material, self.material_opening_radius)

            leftovers -= material # calculate what is left: i.e. the VNC when subtracting iodine

            material /= self.extraction_materials[m].linear_att_coeff(self.case.effective_kev) # divide by mu to get 'a' in unitless concentration

            materials_.append(material)
        
        return materials_, leftovers
    
    def _is_relevant_extraction_organ(self, organ_name, tissue):
        if organ_name.lower() in map(str.lower, self.skip_organ if self.skip_organ is not None else []):
            return False

        if self.skip_tissue_type is not None:
            skip_tissue_types = self.skip_tissue_type
            if not isinstance(skip_tissue_types, (list, tuple)):
                skip_tissue_types = [skip_tissue_types]
            if isinstance(tissue.type, tuple(skip_tissue_types)):
                return False

        if self.skip_unlisted_organs and organ_name.lower() not in self.material_fractions:
            return False

        return True

    def remove_dilated_tissue_type(self, material, type_to_remove=Bone):
        """ Dilate bone masks and remove them from the material map to avoid bone leakage """
        for organ_number, organ_name in self.case.segmentation_names.items():
            tissue = Tissue(organ_name, self.atomic_composition_table)
            print(tissue.type)
            if isinstance(tissue.type, type_to_remove):
                print('Dilating and removing', organ_name, 'from material map')
                bone_map = self.segmentation == organ_number
                kernel = [7] * 3
                #kernel[ImageReader.get_z_index(self.case.CT)] = 0 # set z dilate to 0
                bone_map_expanded = sitk.BinaryDilate(bone_map, kernel)
                bone_map_edge = sitk.Subtract(bone_map_expanded, bone_map) # only keep the dilated part
                possible_bone_pixels = self.case.CT > to_mu(80, self.case.effective_kev) # keep high HU values inside expanded bone mask to refine the mask
                bone_map_edge = sitk.Mask(bone_map_edge, possible_bone_pixels, outsideValue=0)
                bone_map_expanded = sitk.Add(bone_map_edge, bone_map) # add original bone mask to the filtered dilated part
                print('Bone mask before and after dilation:', sitk.GetArrayFromImage(bone_map).sum(), sitk.GetArrayFromImage(bone_map_expanded).sum())
                print('Unique values in bone_map:', np.unique(sitk.GetArrayFromImage(bone_map)))
                # Update segmentation so that the bone mask includes the dilated bone region
                bone_map_expanded = sitk.ConnectedComponent(bone_map_expanded)
                bone_map_expanded = sitk.RelabelComponent(bone_map_expanded, sortByObjectSize=True)
                bone_map_expanded = sitk.Equal(bone_map_expanded, 1)
                #bone_map_expanded = sitk.BinaryMorphologicalOpening(bone_map_expanded, [1]*3)

                bone_map_expanded = sitk.Median(bone_map_expanded, [1]*3)

                self.case.segmentation = sitk.Mask(self.case.segmentation, sitk.Not(bone_map_expanded), outsideValue=organ_number)
                self.segmentation = self.case.segmentation
        material = sitk.Mask(material, sitk.Not(bone_map_expanded), outsideValue=0)
                
        return material

    def open_material_map(self, material, radius):
        """Remove small material islands by opening the positive material support."""
        material_mask = material > 0
        material_mask = sitk.BinaryMorphologicalOpening(material_mask, self._dilation_kernel(radius))
        return sitk.Mask(material, material_mask, outsideValue=0)

    
    def filter_material_map(self, material):
         #material = sitk.Bilateral(material, .15, 10, 10)
        material_mask = material>0
        material = sitk.Mask(material, material_mask, outsideValue=0)
        
        kernel = [0.5] * 3
        if not self.case.continuous_slices:
            kernel[self.case.z_axis_index] = 0 # set z blurring to 0
        material = sitk.DiscreteGaussian(material, variance=kernel)
            
        material = sitk.Mask(material, material_mask, outsideValue=0) # mask the blurred materials to the right part
        return material

    
class ChangeOfBasisExtractor(MaterialExtractor):
    """
    Extracts material basis images from CT data using a change-of-basis approach.
    This extractor computes the local densities of a set of basis materials for each voxel in a CT image,
    based on their linear attenuation coefficients at multiple energies. It uses a pseudo-inverse of the
    attenuation matrix to solve for the material weights, optionally skipping specified atoms or tissue types.

    Parameters
    ----------
    extraction_materials : list[Atom] | list[Tissue]
        The materials (atoms or tissues) to use as the basis for decomposition. Must contain more than one material.
    atomic_composition_table : dict
        Table mapping material names to their atomic compositions.
    skip_atom : str | Atom, optional
        Atom type to skip when computing attenuation coefficients.
    skip_tissue_type : type | tuple[type], optional
        Tissue types to skip during extraction. Voxels of this type are assigned to the first material.

    Attributes
    ----------
    n_materials : int
        Number of basis materials.
    energies : np.ndarray
        Array of energies (in keV) at which attenuation coefficients are evaluated.
    skip_atom : str | Atom
        Atom type to skip.
    extraction_materials : list
        List of basis material objects.
    atomic_composition_table : dict
        Atomic composition table.
    skip_tissue_type : type | tuple[type]
        Tissue type(s) to skip.

    Methods
    -------
    extract(image=None)
        Performs the material decomposition on the provided CT image (or the default case image if not provided).
        Returns a list of SimpleITK images, each representing the local density of a basis material.
    """
        
    def __init__(self, extraction_materials, atomic_composition_table, skip_atom=None, skip_tissue_type=None, skip_organ=None):
        super().__init__(extraction_materials, skip_tissue_type, skip_organ, atomic_composition_table)
        assert isinstance(self.extraction_materials, list), "ChangeOfBasisExtractor.extraction_material must be list of len > 1"
        assert len(self.extraction_materials) > 1, "ChangeOfBasisExtractor.extraction_material must be list of len > 1"
        self.n_materials = len(self.extraction_materials)
        self.energies = np.linspace(20, 140, self.n_materials)
        self.skip_atom = skip_atom
        print(self.skip_tissue_type)

    def extract(self, image=None):
        
        image = self.case.CT if image is None else image

        M = np.zeros((len(self.energies), self.n_materials))
        for m, material in enumerate(self.extraction_materials):
            M[:, m] = material.linear_att_coeff(self.energies)

        effective_mu = [material.linear_att_coeff(self.case.effective_kev) for material in self.extraction_materials]

        ct = sitk.GetArrayFromImage(image)
        segmentation = sitk.GetArrayFromImage(self.segmentation)
                
        M_inv = np.linalg.pinv(M)
        local_densities = np.zeros((self.n_materials, *ct.shape))

        for organ_number, organ_name in self.case.segmentation_names.items():
            tissue = Tissue(organ_name, self.atomic_composition_table)

            if self.skip_tissue_type is not None and isinstance(tissue.type, tuple([self.skip_tissue_type])):
                print('Skipping', organ_name, tissue.type)
                w = np.array([1, 0]) # set all weight to first material (water in most cases). Should be updated with a separate kwarg
            else:
                mu = tissue.linear_att_coeff(self.energies, skip_atom=self.skip_atom)
                w = M_inv @ mu
                print(organ_name, tissue.type)

            organ_local_density = segmentation == organ_number # should be updated to allow fractions

            # correction factor to material weights to align with input image
            w_corr = ct[organ_local_density] / np.dot(w, effective_mu)

            w = w[:, np.newaxis] * w_corr[np.newaxis, :]

            local_densities[:, organ_local_density==1] = w

        materials = []
        for i in range(self.n_materials):
            mat = local_densities[i,...]
            mat = sitk.GetImageFromArray(mat)
            mat.CopyInformation(image)
            materials.append(mat)

        return materials





if __name__ == '__main__':
    from tissue import Bone, Fat
    from case import Case
    import argparse, os
    from utils import to_HU
    from pathlib import Path
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import torch
    from image_reader import ImageReader
   
    import os

    parser = argparse.ArgumentParser(description='Material extraction from CT images.')
    parser.add_argument('--datadir', type=str, default='/mnt/data0/kits23/', help='Directory containing dataset')
    parser.add_argument('--ct_filename', type=str, help='The filename of the input ct file', default=CT_FILENAME)    
    parser.add_argument('--savedir', type=str, default=None, help='Directory to save results')
    parser.add_argument('--cases', type=str, default='0', help='Slice or list of cases to process, e.g. "0:10" or "1,2,3"')
    parser.add_argument('--plot', action='store_true', help='Plot results')
    args = parser.parse_args()

    datadir = args.datadir
    savepath = args.savedir if args.savedir is not None else datadir

    subdirs = sorted([d for d in os.listdir(datadir) if 'case' in d and Path(datadir, d).is_dir()]) # sort dirs
    
    print(len(subdirs))
    if ':' in args.cases:
        start, end = args.cases.split(':')
        cases = slice(int(start) if start else None, int(end) if end else None)
    elif ',' in args.cases:
        cases = [int(x) for x in args.cases.split(',')]
    else:
        cases = slice(int(args.cases), int(args.cases)+1)

    subdirs = subdirs[cases] # extract cases of interest

    plot = args.plot

    for case_num in tqdm(subdirs, desc='Extracting materials from cases...'):
        print('Processing', case_num)

        subdir = Path(datadir, case_num)
        savedir = Path(savepath, case_num)

        os.makedirs(savedir, exist_ok=True)

        case = Case(subdir, ct_filename=args.ct_filename, read_midslice_only=False, read_slices=None)#, crop_to_organ=['kidney_left', 'kidney_right'])
        print(case.CT.GetSize(), case.CT.GetSpacing(), case.CT.GetOrigin(), case.CT.GetDirection())
        atomicCompTable = TissueAtomicCompositionTable()
        subtractionExtractor = SubtractionExtractor([Atom('I', density=0.001)], 
                                                    atomicCompTable, 
                                                    clip_values=(0, None), 
                                                    skip_tissue_type=[Bone, Fat],
                                                    skip_organ=['cartilage', 'lobe_lung_left', 'lobe_lung_right', 'lung_lower_lobe_right', 'lung_lower_lobe_left'],
                                                    filter_result=True, 
                                                    dilated_tissue_type_to_remove=[Bone],
                                                    refine_bone_masks=True,
                                                    bone_refinement_hu_threshold=80,
                                                    bone_refinement_opening_radius=3,
                                                    bone_refinement_closing_radius=1,
                                                    bone_refinement_distance_smoothing_sigma=0,
                                                    material_opening_radius=1,
                                                    )
        subtractionExtractor.set_case(case)
        subtracted_materials, vnc = subtractionExtractor.extract()
        
        twoBasisExtractor = ChangeOfBasisExtractor([Tissue('water', density=0.001), Atom('Ca', density=0.001)], atomicCompTable)
        twoBasisExtractor.set_case(case)
        twoBasis_materials = twoBasisExtractor.extract(vnc)

        for i, mat in enumerate(subtracted_materials):
            sitk.WriteImage(mat, Path(savedir, f'{subtractionExtractor.extraction_materials[i].name.lower()}.nii.gz'))
        for i, mat in enumerate(twoBasis_materials):
            sitk.WriteImage(mat, Path(savedir, f'{twoBasisExtractor.extraction_materials[i].name.lower()}.nii.gz'))

        sitk.WriteImage(case.segmentation, Path(savedir, 'segmentation.nii.gz'))
        torch.save({'segmentation': case.segmentation_names, 'body': case.body_names, 'indices': case.organ_fov}, Path(savedir, 'labels.pt'))


        if plot:
            # Plot the result to the subdir
            print('Plotting results...')

            z_index = ImageReader.get_z_index(case.CT)
            slice_number = case.CT.GetSize()[z_index] // 2
            slice_obj = [slice(None)] * 3
            slice_obj[z_index] = slice_number
        
            images = [sitk.GetArrayFromImage(case.CT[slice_obj]), sitk.GetArrayFromImage(case.segmentation[slice_obj])]
            titles = ['Input CT', 'Segmentation']

            for i, mat in enumerate(subtracted_materials):
                im = sitk.GetArrayFromImage(mat[slice_obj])
                print(im.shape)
                images.append(im)
                titles.append(f'{subtractionExtractor.extraction_materials[i].name} {subtractionExtractor.extraction_materials[i].density:.2g} g/cm³')
            
            for i, mat in enumerate(twoBasis_materials):
                im = sitk.GetArrayFromImage(mat[slice_obj])
                images.append(im)
                titles.append(f'{twoBasisExtractor.extraction_materials[i].name} {twoBasisExtractor.extraction_materials[i].density:.2g} g/cm³')
    

            n_images = len(images)
            n_cols = int(np.ceil(np.sqrt(n_images)))
            n_rows = int(np.ceil(n_images / n_cols))
            fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharex=True, sharey=True)
            axs=axs.flatten()
            print(axs.shape)
            clims =[[None, None], [None, None], [0,10], [0,1000], [0,150]] 
            for i, im in enumerate(images):
                im=axs[i].imshow(im.squeeze().T, cmap='gray', clim=clims[i])
                axs[i].set_title(titles[i])
                plt.colorbar(im, ax=axs[i], fraction=0.046, pad=0.04)
                
            for ax in axs: ax.axis('off')
            #plt.tight_layout()
            plt.savefig(Path(savedir, f'spectral_phantom_result.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
