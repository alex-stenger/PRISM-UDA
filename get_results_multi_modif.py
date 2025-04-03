import os
import cv2
import numpy as np
import argparse

def calculate_iou(pred_mask, gt_mask, class_value):
    pred_class = (pred_mask == class_value).astype(np.uint8)
    gt_class = (gt_mask == class_value).astype(np.uint8)
    
    intersection = np.logical_and(pred_class, gt_class)
    union = np.logical_or(pred_class, gt_class)
    
    iou = (np.sum(intersection) + 1e-6) / (np.sum(union) + 1e-6)
    return iou

def test(pred_path, gt_path):
    iou_mitochondria = 0
    iou_reticulum = 0
    total_samples = 0
    
    for gt_file in os.listdir(gt_path):
        if not gt_file.endswith('.png'):
            continue

        if "labelTrainIds" not in gt_file:
            gt_img = cv2.imread(os.path.join(gt_path, gt_file), cv2.IMREAD_GRAYSCALE)
            pred = cv2.imread(os.path.join(pred_path, gt_file), cv2.IMREAD_GRAYSCALE)

            if np.count_nonzero(pred==70)!=0 and np.count_nonzero(pred==119)!=0 and np.count_nonzero(pred==90)!=0 :
            
                pred[pred == 70] = 255  # mitochondria
                pred[pred == 119] = 128  # reticulum
                pred[pred == 90] = 0  # background
                
                iou_mito = calculate_iou(pred, gt_img, 255)  # mitochondria IoU
                iou_retic = calculate_iou(pred, gt_img, 128)  # reticulum IoU
                
                print(f"GT: {gt_file}, IoU Mitochondria: {iou_mito:.4f}, IoU Reticulum: {iou_retic:.4f}")
                
                iou_mitochondria += iou_mito
                iou_reticulum += iou_retic
                total_samples += 1
    
    if total_samples > 0:
        mean_iou_mitochondria = iou_mitochondria / total_samples
        mean_iou_reticulum = iou_reticulum / total_samples
        mean_iou = (mean_iou_mitochondria + mean_iou_reticulum) / 2
        
        print(f"Mean IoU Mitochondria: {mean_iou_mitochondria:.4f}")
        print(f"Mean IoU Reticulum: {mean_iou_reticulum:.4f}")
        print(f"Mean IoU (Overall): {mean_iou:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate IoU per class and mean IoU.")
    parser.add_argument("--pred_path", type=str, help="Path to the directory containing predicted masks.")
    parser.add_argument("--gt_path", type=str, help="Path to the directory containing ground truth masks.")
    args = parser.parse_args()
    
    test(args.pred_path, args.gt_path)
