import nibabel
import os
import scipy.io
import nipype.interfaces.spm as spm
import nipype.interfaces.spm.utils as spmu
import numpy as np
from skimage.restoration import denoise_nl_means, estimate_sigma
from nilearn import plotting

# matlab_cmd = '/i2bm/local/cat12-standalone/run_spm12.sh /i2bm/local/cat12-standalone/mcr/v93/ script'
matlab_cmd = '/i2bm/local/spm12-standalone/run_spm12.sh /i2bm/local/spm12-standalone/mcr/v713/ script'
# matlab_cmd = '/neurospin/local/bin/matlab-R2019a'
spm.SPMCommand.set_mlab_paths(matlab_cmd=matlab_cmd, use_mcr=True) 


def denoising_nlm(image, output):
    image = nibabel.load(image)
    arr_image = image.get_fdata()
    sigma_est = np.mean(estimate_sigma(arr_image))
    arr_denoised_image = denoise_nl_means(arr_image, h=1.15 * sigma_est,
                                          patch_size=9,
                                          patch_distance=5)
    denoised_image = nibabel.Nifti1Image(arr_denoised_image, image.affine)
    nibabel.save(denoised_image, output)


def coregistration(target_file, moving_file, coreg_path):
    coreg = spmu.CalcCoregAffine()
    coreg.inputs.target = target_file
    coreg.inputs.moving = moving_file
    coreg.inputs.mat = coreg_path
    coreg.run()


def apply_defor(defor, moving_file):
    # apply H coil to MNI
    norm12 = spm.Normalize12()
    norm12.inputs.apply_to_files = moving_file
    norm12.inputs.deformation_file = defor
    norm12.inputs.write_bounding_box = [[-78, -112, -70], [78, 76, 85]]
    norm12.inputs.write_voxel_sizes = [1, 1, 1]
    norm12.inputs.write_interp = 4
    norm12.inputs.jobtype = 'write'
    norm12.inputs.use_mcr = True
    # norm12.inputs.out_prefix = "MNI_"
    norm12.run()


def apply_transform(input, matrix, output):
    applymat = spmu.ApplyTransform()
    applymat.inputs.in_file = input
    applymat.inputs.mat = matrix  # .mat
    applymat.inputs.out_file = output
    applymat.run()


def dicom_to_nii(liste_filename):
    di = spmu.DicomImport()
    di.inputs.in_files = liste_filename
    di.run()


def write_matlabbatch(template, nii_file, tpm_file, darteltpm_file, outfile):
    """ Complete matlab batch from template.

    Parameters
    ----------
    template: str
        path to template batch to be completed.
    nii_files: list
        the Nifti image to be processed.
    tpm_file: str
        path to the SPM TPM file.
    darteltpm_file: str
        path to the CAT12 tempalte file.
    outfile: str
        path to the generated matlab batch file that can be used to launch
        CAT12 VBM preprocessing.
    """
    nii_file_str=""
    for i in nii_file:
        nii_file_str += "'{0}' \n".format(ungzip_file(i, 
                                        outdir=os.path.dirname(outfile)))
    with open(template, "r") as of:
        stream = of.read()
    stream = stream.format(anat_file=nii_file_str, tpm_file=tpm_file,
                        darteltpm_file=darteltpm_file)
    with open(outfile, "w") as of:
        of.write(stream)


def plot_anat_Li(li_mni_denoised, anat_MNI, threshold):
    li_img = nibabel.load(li_mni_denoised)
    bg_img = nibabel.load(anat_MNI)

    arr = li_img.get_fdata()
    arr[arr < threshold] = 0
    li_img = nibabel.Nifti1Image(arr, li_img.affine)
    plotting.plot_stat_map(li_img, bg_img, cut_coords=(-35, 54, -44),
                           cmap=plotting.cm.black_blue)
    plotting.show()


def find_threshold(mask_MNI, li_mni_denoised):
    li_img = nibabel.load(li_mni_denoised)
    mask_img = nibabel.load(mask_MNI)
    arr = li_img.get_fdata()
    arr[mask_img.get_fdata() != 0] = 0
    threshold = np.max(arr)
    print(threshold)
    return threshold


# target_file = "/volatile/7Li/data_test/08_09_21/data/anat_7T/t1_mpr_tra_iso1_0mm.nii"
# moving_file = "/volatile/7Li/data_test/08_09_21/data/Li/trufi_7Li_RR.nii"
# coreg_path = "/volatile/7Li/data_test/08_09_21/transfo/Li_to_Lianat.mat"

# target_file = "/volatile/7Li/data_test/08_09_21/data/anat_3T/t1_weighted_sagittal_1_0iso.nii"
# moving_file = "/volatile/7Li/data_test/08_09_21/data/anat_7T/t1_mpr_tra_iso1_0mm.nii"
# coreg_path = "/volatile/7Li/data_test/08_09_21/transfo/anat7T_to_anat3T.mat"

# target_file = "/volatile/7Li/data_test/08_09_21/data_rlink/anat_3TLi/01004RL20210430M033DT1Li_noPN_DIS2D_S007.nii"
# moving_file = "/volatile/7Li/data_test/08_09_21/data_rlink/Li/01004RL20210430M03trufi_S005.nii"
# coreg_path = "/volatile/7Li/data_test/08_09_21/transfo_rlink/Li_to_Lianat.mat"

# target_file = "/volatile/7Li/data_test/08_09_21/data_rlink/anat_3T/01004RL20210430M033DT1_noPN_DIS2D_S009.nii"
# moving_file = "/volatile/7Li/data_test/08_09_21/data_rlink/anat_3TLi/01004RL20210430M033DT1Li_noPN_DIS2D_S007.nii"
# coreg_path = "/volatile/7Li/data_test/08_09_21/transfo_rlink/anat3TLi_to_anat3T.mat"

# coregistration(target_file, moving_file, coreg_path)

# defor = "/volatile/7Li/data_test/08_09_21/transfo_rlink/y_3T_to_MNI.nii"
# moving_file = "/volatile/7Li/data_test/08_09_21/data_rlink/anat_3T/01004RL20210430M033DT1_noPN_DIS2D_S009.nii"
# apply_defor(defor, moving_file)

# input = "/volatile/7Li/data_test/08_09_21/data/anat_7T/t1_mpr_tra_iso1_0mm.nii"
# matrix = "/volatile/7Li/data_test/08_09_21/transfo/anat7T_to_anat3T.mat"
# output = "/volatile/7Li/data_test/08_09_21/transfo/anat_7T3T.nii"
# apply_transform(input, matrix, output)
################################################################################
# Li_bl = nibabel.load("/volatile/7Li/data_test/08_09_21/data/Li/trufi_7Li_RR.nii")

# trans7T = scipy.io.loadmat("/volatile/7Li/data_test/08_09_21/transfo/inverse_Li_to_Lianat.mat")
# trans7T_mat = trans7T['M']

# trans3T = scipy.io.loadmat("/volatile/7Li/data_test/08_09_21/transfo/inverse_anat7T_to_anat3T.mat")
# trans3T_mat = trans3T['M']

# mixte_mat = np.dot(trans7T_mat, trans3T_mat)

# deformfiel_nl = "/volatile/7Li/data_test/08_09_21/transfo/y_3T_to_MNI.nii"
# img = nibabel.load(deformfiel_nl)
# new_affine = np.dot(mixte_mat, img.affine)
# normalized = nibabel.Nifti1Image(img.get_fdata(), new_affine)
# nibabel.save(normalized, "/volatile/7Li/data_test/08_09_21/transfo/y_combine.nii")

# defor = "/volatile/7Li/data_test/08_09_21/transfo/y_combine.nii"
# moving_file = "/volatile/7Li/data_test/08_09_21/data/Li/trufi_7Li_RR.nii"
##################
# Li_bl = nibabel.load("/volatile/7Li/data_test/08_09_21/data_rlink/Li/01004RL20210430M03trufi_S005.nii")
# trans7T = scipy.io.loadmat("/volatile/7Li/data_test/08_09_21/transfo_rlink/inverse_Li_to_Lianat.mat")
# trans7T_mat = trans7T['M']
# trans3T = scipy.io.loadmat("/volatile/7Li/data_test/08_09_21/transfo_rlink/inverse_anat3TLi_to_anat3T.mat")
# trans3T_mat = trans3T['M']
# mixte_mat = np.dot(trans7T_mat, trans3T_mat)
# deformfiel_nl = "/volatile/7Li/data_test/08_09_21/transfo_rlink/y_3T_to_MNI.nii"
# img = nibabel.load(deformfiel_nl)
# new_affine = np.dot(mixte_mat, img.affine)
# normalized = nibabel.Nifti1Image(img.get_fdata(), new_affine)
# nibabel.save(normalized, "/volatile/7Li/data_test/08_09_21/transfo_rlink/y_combine.nii")

# defor = "/volatile/7Li/data_test/08_09_21/transfo_rlink/y_combine.nii"
# moving_file = "/volatile/7Li/data_test/08_09_21/data_rlink/Li/01004RL20210430M03trufi_S005.nii"
# #####################
# apply_defor(defor, moving_file)


###############################################################################
# from pymialsrtk.interfaces.preprocess import BtkNLMDenoising

# nlmDenoise = BtkNLMDenoising()
# nlmDenoise.inputs.bids_dir = "/volatile/7Li/data_test/08_09_21/transfo"
# nlmDenoise.inputs.in_file = "wtrufi_7Li_RR.nii"
# nlmDenoise.inputs.in_mask = "p0t1_weighted_sagittal_1_0iso.nii"
# nlmDenoise.inputs.weight = 0.2
# nlmDenoise.run() # doctest: +SKIP


def pipeline_lithium(target_anatLi, target_anat, moving_file_Li,
                     transfo_folder,
                     executable_cat12, standalone_cat12,
                     mcr_matlab, matlabbatch, tpm_file, darteltpm_file):
    # # Li to anat Li
    # coreg_path_Li = os.path.join(transfo_folder, "Li_to_Lianat.mat")
    # coregistration(target_anatLi, moving_file_Li, coreg_path_Li)
    # # anat Li to anat
    # coreg_path_anat = os.path.join(transfo_folder, "anatLi_to_anat.mat")
    # coregistration(target_anat, target_anatLi, coreg_path_anat)
    # # write matlabbatch
    # write_matlabbatch(matlabbatch, target_anat, tpm_file, darteltpm_file,
    #                   transfo_folder)
    # # anat to MNI
    # subprocess.check_call([executable_cat12,
    #                        "-s", standalone_cat12,
    #                        "-m", mcr_matlab,
    #                        "-b", matlabbatch])
    # # create combine transformation
    # trans_Lianat = scipy.io.loadmat(os.path.join(transfo_folder,
    #                                 "inverse_Li_to_Lianat.mat"))
    # trans_Lianat_mat = trans_Lianat['M']

    # trans_anat = scipy.io.loadmat(os.path.join(transfo_folder,
    #                               "inverse_anatLi_to_anat.mat"))
    # trans_anat_mat = trans_anat['M']

    # mixte_mat = np.dot(trans_Lianat_mat, trans_anat_mat)
    # deformfiel_nl = target_anat.split(os.sep).insert(-1, "mri")
    # deformfiel_nl[-1] = "y_{0}".format(deformfiel_nl[-1]) 
    # deformfiel_nl = os.sep.join(deformfiel_nl)
    # img = nibabel.load(deformfiel_nl)
    # new_affine = np.dot(mixte_mat, img.affine)
    # normalized = nibabel.Nifti1Image(img.get_fdata(), new_affine)
    # nibabel.save(normalized, os.path.join(transfo_folder, "y_combine.nii"))

    defor = os.path.join(transfo_folder, "y_combine.nii")
    # apply_defor(defor, moving_file_Li)
   
    # denoising
    Li_MNI = moving_file_Li.split(os.sep)
    anat_MNI = target_anat.split(os.sep)
    anat_MNI.insert(-1, "mri")
    mask_path = anat_MNI.copy()
    mask_path[-1] = "p0{0}".format(mask_path[-1])
    mask_path = os.sep.join(mask_path)
    apply_defor(defor, mask_path)
    mask_path_MNI = anat_MNI.copy()
    mask_path_MNI[-1] = "wp0{0}".format(mask_path_MNI[-1])
    mask_path_MNI = os.sep.join(mask_path_MNI)
    Li_MNI_denoised = Li_MNI.copy()
    Li_MNI_denoised[-1] = "w{0}_denoised{1}".format(
                                    os.path.splitext(Li_MNI_denoised[-1])[0],
                                    os.path.splitext(Li_MNI_denoised[-1])[1])
    Li_MNI_denoised = os.sep.join(Li_MNI_denoised)
    Li_MNI[-1] = "w{0}".format(Li_MNI[-1])
    Li_MNI = os.sep.join(Li_MNI)
    denoising_nlm(Li_MNI, Li_MNI_denoised)

    # find threshold
    threshold = find_threshold(mask_path_MNI, Li_MNI_denoised)
        
    # plot results
    anat_MNI = target_anat.split(os.sep)
    anat_MNI[-1] = "w{0}".format(anat_MNI[-1])
    anat_MNI = os.sep.join(anat_MNI)
    plot_anat_Li(Li_MNI_denoised, anat_MNI, threshold)



# initialization

target_anatLi = "/volatile/7Li/data_test/08_09_21/data_rlink/anat_3TLi/01004RL20210430M033DT1Li_noPN_DIS2D_S007.nii"
target_anat = "/volatile/7Li/data_test/08_09_21/data_rlink/anat_3T/01004RL20210430M033DT1_noPN_DIS2D_S009.nii"
moving_file_Li = "/volatile/7Li/data_test/08_09_21/data_rlink/Li/01004RL20210430M03trufi_S005.nii"
transfo_folder = "/volatile/7Li/data_test/08_09_21/transfo_rlink"
executable_cat12 = "/i2bm/local/cat12-standalone/standalone/cat_standalone.sh"
standalone_cat12 = "/i2bm/local/cat12-standalone"
mcr_matlab = "/i2bm/local/cat12-standalone/mcr/v93"
matlabbatch = "/volatile/7Li/Limri/Limri/resources/cat12vbm_matlabbatch.m"
tpm_file = "/volatile/BRAINPREP/cat12/CAT12.7_r1743_R2017b_MCR_Linux/spm12_mcr/home/gaser/gaser/spm/spm12/tpm/TPM.nii"
darteltpm_file = "/volatile/BRAINPREP/cat12/CAT12.7_r1743_R2017b_MCR_Linux/spm12_mcr/home/gaser/gaser/spm/spm12/toolbox/cat12/templates_volumes/Template_1_IXI555_MNI152.nii"



pipeline_lithium(target_anatLi, target_anat, moving_file_Li,
                 transfo_folder,
                 executable_cat12, standalone_cat12,
                 mcr_matlab, matlabbatch, tpm_file, darteltpm_file)
