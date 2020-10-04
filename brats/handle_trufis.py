import nibabel as nib
import os
from fetal_net.metrics import dice_coefficient_np, vod_coefficient_np, get_surface_distances, \
    false_negative_rate, false_positive_rate
from fetal_net.postprocess import postprocess_prediction
import glob


pred_dir = r"/mnt/trufi_results"
sub_ids = os.listdir(pred_dir)
for sub_id in sub_ids:
    print(sub_id)
    if os.path.exists(os.path.join(pred_dir, sub_id, 'prediction.nii.gz')):
        pred = nib.load(os.path.join(pred_dir, sub_id, 'prediction.nii.gz'))
        pred_data = pred.get_data()
        binary_pred = postprocess_prediction(pred_data)
        nib.save(nib.Nifti1Image(binary_pred, pred.affine, header=pred.header), os.path.join(pred_dir, sub_id, 'binary_prediction.nii'))
        os.remove(os.path.join(pred_dir, sub_id, 'prediction.nii.gz'))
    