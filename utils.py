from tissue import Tissue
import SimpleITK as sitk
import numpy as np

def to_mu(hounsfield_unit, kev=70):
    mu_water = Tissue('water').linear_att_coeff(kev)
    return hounsfield_unit * mu_water / 1000.0 + mu_water

def to_HU(mu, kev=70):
    mu_water = Tissue('water').linear_att_coeff(kev)
    return (mu - mu_water) * 1000.0 / mu_water

def bilateral_2d_slices(image, domainSigma=4.0, rangeSigma=50.0, numberOfRangeGaussianSamples=100, axis=0):
    """
    Applies bilateral filtering only in 2D slices along a specified axis.
    """
    # Convert SimpleITK Image to NumPy array
    img_array = sitk.GetArrayFromImage(image)  # numpy shape: (Z, Y, X)
    
    # Apply bilateral filtering to each 2D slice in the selected axis
    for i in range(img_array.shape[axis]):
        if axis == 0:
            img_array[i, :, :] = sitk.GetArrayFromImage(
                sitk.Bilateral(sitk.GetImageFromArray(img_array[i, :, :]), domainSigma, rangeSigma, numberOfRangeGaussianSamples)
            )
        elif axis == 1:
            img_array[:, i, :] = sitk.GetArrayFromImage(
                sitk.Bilateral(sitk.GetImageFromArray(img_array[:, i, :]), domainSigma, rangeSigma, numberOfRangeGaussianSamples)
            )
        elif axis == 2:
            img_array[:, :, i] = sitk.GetArrayFromImage(
                sitk.Bilateral(sitk.GetImageFromArray(img_array[:, :, i]), domainSigma, rangeSigma, numberOfRangeGaussianSamples)
            )

    # Convert back to SimpleITK Image
    filtered_image = sitk.GetImageFromArray(img_array)
    filtered_image.CopyInformation(image)
    return filtered_image


def resample(image, target_spacing, target_size, **kwargs):
    """
    Resamples a CT image to a fixed axial resolution while ensuring a 512x512 in-plane size.
    Also centers the volume in the final FOV.

    Args:
        image (sitk.Image): Input SimpleITK image (assumed shape ZxXxY).
        target_spacing (tuple): Desired spacing (z, x, y). If z is None, it remains unchanged.
        target_size (tuple): Desired output size (depth, width, height). If depth is None, it is computed automatically.

    Returns:
        sitk.Image: Resampled and centered image.
    """
    # Get original properties
    print(image.GetSize(), image.GetSpacing())
    orig_size = np.array(image.GetSize())  # (Z, X, Y)
    orig_spacing = np.array(image.GetSpacing())  # (Z, X, Y) spacing
    orig_origin = np.array(image.GetOrigin())
    orig_direction = image.GetDirection()

    # Set target spacing (keeping original Z spacing if None)
    if target_spacing[0] is None:
        target_spacing = (orig_spacing[0], target_spacing[1], target_spacing[2])

    # Compute new Z size to preserve FOV
    if target_size[0] is None:
        target_size = (
            int(round(orig_size[0] * (orig_spacing[0] / target_spacing[0]))),  # Compute new depth
            target_size[1],  # Keep X as fixed 512
            target_size[2]   # Keep Y as fixed 512
        )

    # Set up the resampler
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(target_size)
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetOutputDirection(orig_direction)
    resampler.SetInterpolator(sitk.sitkLinear)  # Use nearest neighbor for segmentation masks
    resampler.SetDefaultPixelValue(0)  # Background value
    resampler.SetTransform(sitk.Transform())

    # Compute new origin to center the volume
    new_origin = orig_origin + (orig_size * orig_spacing - np.array(target_size) * np.array(target_spacing)) / 2.0
    resampler.SetOutputOrigin(tuple(new_origin))

    # Perform resampling
    resampled_image = resampler.Execute(image)
    print(resampled_image.GetSize(), resampled_image.GetSpacing())

    return resampled_image
