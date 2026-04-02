import os
import random
from PIL import Image, ImageEnhance
import numpy as np
from tqdm import tqdm
import argparse

def augment_and_save(img_A_path, img_B_path, gt_path, aug_num=1):
    imgA = Image.open(img_A_path).convert('RGB')
    imgB = Image.open(img_B_path).convert('RGB')
    gt = Image.open(gt_path).convert('L')
    
    base_name_A = os.path.basename(img_A_path)
    base_name_B = os.path.basename(img_B_path)
    base_name_gt = os.path.basename(gt_path)
    
    dir_A = os.path.dirname(img_A_path)
    dir_B = os.path.dirname(img_B_path)
    dir_gt = os.path.dirname(gt_path)

    for i in range(1, aug_num + 1):
        outA, outB, outGt = imgA.copy(), imgB.copy(), gt.copy()
        
        # 1. Flip Horizontal/Vertical
        if random.random() > 0.5:
            outA = outA.transpose(Image.FLIP_LEFT_RIGHT)
            outB = outB.transpose(Image.FLIP_LEFT_RIGHT)
            outGt = outGt.transpose(Image.FLIP_LEFT_RIGHT)
            
        if random.random() > 0.5:
            outA = outA.transpose(Image.FLIP_TOP_BOTTOM)
            outB = outB.transpose(Image.FLIP_TOP_BOTTOM)
            outGt = outGt.transpose(Image.FLIP_TOP_BOTTOM)
            
        # 2. Rotations Exactes (90, 180, 270)
        # Ceci évite les bordures noires d'une rotation libre
        rot = random.choice([None, Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270])
        if rot is not None:
            outA = outA.transpose(rot)
            outB = outB.transpose(rot)
            outGt = outGt.transpose(rot)
            
        # 3. Bruit Gaussien
        if random.random() > 0.6:
            noiseA = np.random.normal(0, 15.0, np.array(outA).shape)
            outA = Image.fromarray(np.clip(np.array(outA, dtype=float) + noiseA, 0, 255).astype(np.uint8))
            
            noiseB = np.random.normal(0, 15.0, np.array(outB).shape)
            outB = Image.fromarray(np.clip(np.array(outB, dtype=float) + noiseB, 0, 255).astype(np.uint8))
            
        # 4. Modifications des contrastes et luminosité (indépendantes du masque)
        if random.random() > 0.5:
            bright = random.uniform(0.8, 1.2)
            outA = ImageEnhance.Brightness(outA).enhance(bright)
            outB = ImageEnhance.Brightness(outB).enhance(bright)
            
            contrast = random.uniform(0.8, 1.2)
            outA = ImageEnhance.Contrast(outA).enhance(contrast)
            outB = ImageEnhance.Contrast(outB).enhance(contrast)
            
        # Sauvegarde avec extension suffixée
        name_no_ext, ext = os.path.splitext(base_name_A)
        new_name_A = f"{name_no_ext}_aug{i}{ext}"
        
        name_no_ext_B, ext_B = os.path.splitext(base_name_B)
        new_name_B = f"{name_no_ext_B}_aug{i}{ext_B}"
        
        name_no_ext_gt, ext_gt = os.path.splitext(base_name_gt)
        new_name_gt = f"{name_no_ext_gt}_aug{i}{ext_gt}"
        
        outA.save(os.path.join(dir_A, new_name_A))
        outB.save(os.path.join(dir_B, new_name_B))
        outGt.save(os.path.join(dir_gt, new_name_gt))

def main():
    parser = argparse.ArgumentParser(description="Script d'augmentation de données hors-ligne (Offline Data Augmentation)")
    parser.add_argument('--dataset_path', type=str, default='./data/LEVIR-CD/train/', help='Chemin du dossier train')
    parser.add_argument('--aug_num', type=int, default=3, help='Nombre d\'images générées par image originale')
    args = parser.parse_args()

    path_A = os.path.join(args.dataset_path, 'A')
    path_B = os.path.join(args.dataset_path, 'B')
    path_gt = os.path.join(args.dataset_path, 'label')
    
    if not os.path.exists(path_A) or not os.path.exists(path_B) or not os.path.exists(path_gt):
        print(f"Erreur : Impossible de trouver un des dossiers A, B, ou label dans {args.dataset_path}")
        return
        
    # Exclure les images qui comportent déjà '_aug' pour éviter de boucler à l'infini si relancé
    filenames = [f for f in os.listdir(path_A) if f.endswith(('.png', '.jpg')) and '_aug' not in f]
    print(f"Images originales trouvées: {len(filenames)}.")
    print(f"Génération de {args.aug_num} images au hasard pour chaque...")
    
    for fname in tqdm(filenames):
        img_A_path = os.path.join(path_A, fname)
        img_B_path = os.path.join(path_B, fname)
        gt_path = os.path.join(path_gt, fname)
        
        if os.path.exists(img_B_path) and os.path.exists(gt_path):
            try:
                augment_and_save(img_A_path, img_B_path, gt_path, aug_num=args.aug_num)
            except Exception as e:
                print(f"Erreur en traitant {fname}: {e}")
        else:
            print(f"Le fichier {fname} est manquant dans B ou label.")
            
    # Recapitulatif
    total_imgs = len([f for f in os.listdir(path_A) if f.endswith(('.png', '.jpg'))])
    print("\n✅ Augmentation terminée avec succès !")
    print(f"La base de d'entrainement possède maintenant un total de {total_imgs} images.")

if __name__ == '__main__':
    main()
