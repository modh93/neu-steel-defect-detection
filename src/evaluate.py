import argparse
import glob
from ultralytics import YOLO


def evaluate(model_path: str, data_config: str) -> dict:
    """Lance la validation complète et retourne les métriques par classe."""
    model = YOLO(model_path)
    results = model.val(data=data_config)
    return results


def find_false_negatives(model_path: str, images_dir: str, class_name: str, conf: float = 0.25) -> list[str]:
    """
    Retourne la liste des images contenant `class_name` dans leur nom de fichier
    pour lesquelles le modèle n'a prédit aucune box.
    """
    model = YOLO(model_path)
    class_images = glob.glob(f"{images_dir}/{class_name}_*.jpg")

    if not class_images:
        print(f"Aucune image trouvée pour la classe '{class_name}' dans {images_dir}")
        return []

    results = model.predict(class_images, conf=conf, verbose=False)

    false_negatives = [
        img_path for img_path, result in zip(class_images, results)
        if len(result.boxes) == 0
    ]

    return false_negatives


def parse_args():
    parser = argparse.ArgumentParser(description="Évaluation du modèle YOLOv8 sur NEU-DET")
    parser.add_argument("--model", type=str, required=True, help="Chemin vers les poids du modèle (.pt)")
    parser.add_argument("--data", type=str, default="configs/data.yaml", help="Chemin vers data.yaml")
    parser.add_argument("--check-class", type=str, default=None, help="Nom de classe pour l'analyse de faux négatifs")
    parser.add_argument("--images-dir", type=str, default="data/neu-yolo/images/validation")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("=== Métriques globales et par classe ===")
    evaluate(args.model, args.data)

    if args.check_class:
        print(f"\n=== Faux négatifs pour la classe '{args.check_class}' ===")
        fns = find_false_negatives(args.model, args.images_dir, args.check_class)
        print(f"{len(fns)} faux négatifs trouvés :")
        for path in fns:
            print(f"  - {path}")