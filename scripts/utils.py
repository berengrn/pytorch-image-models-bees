import os
import glob
from PIL import Image
import shutil
import csv

def load_classnames(filepath):
    """
    Load classnames from a file.

    Args:
        filepath (string): filepath
    """
    classes = []
    with open(filepath, 'r') as f:
        data = f.readlines()
        for line in data:
            classes.append(line.replace('\n', ''))
    return classes

def read_csv(filepath):
    """
    Read csv.

    Args:
        filepath (string): filepath
    """
    lines = []
    with open(filepath, newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            lines.append(row)
    return lines

def get_csv_header(filepath):
    """
    Get csv header.

    Args:
        filepath (string): filepath
    """
    with open(filepath, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  # Read the first row
    return header

def get_parents(hierarchy_lines):
    """
    Get parents for each node.

    Args:
        hierarchy_lines (string[]): lines of the hierarchy csv
    """
    parents = {}
    for line in hierarchy_lines:
        for i, node in enumerate(line[1:]):
            parents[node] = line[i]
    return parents

def get_taxon_level(hierarchy_lines):
    """
    Get taxonomy level.

    Args:
        hierarchy_lines (string[]): lines of the hierarchy csv
    """
    taxon_levels = {}
    for line in hierarchy_lines[1:]:
        for i, node in enumerate(line):
            taxon_levels[node] = hierarchy_lines[0][i]
    return taxon_levels

def keep_only_n_images(root_folder, n=25):
    """
    Keep only n images from folders inside a root folder.

    Args:
        root_folder (string): dirpath to the root folder of the classes
        n (int): number of images to keep for each classes
    """
    for subdir in os.listdir(root_folder):
        subdir_path = os.path.join(root_folder, subdir)
        
        if os.path.isdir(subdir_path):
            image_files = sorted(glob.glob(os.path.join(subdir_path, "*.*")))
            image_files = [f for f in image_files if f.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".bmp"))]
            
            if len(image_files) > n:
                for img in image_files[n:]:
                    os.remove(img)
            else:
                pass
                print(f"No images found in {subdir_path}")

def remove_classes(root_folder, classes_to_remove):
    """
    Remove classes from a dataset.

    Args:
        root_folder (string): dirpath to the root folder of the classes
        classes_to_remove (string[]): classes to remove
    """
    for class_ in classes_to_remove:
        class_folder = os.path.join(root_folder, class_)
        shutil.rmtree(class_folder)

def copy_folder(src, dst):
    """
    Copy a folder into another folder.

    Args:
        src (string): source folder
        dst (string): destination folder
    """
    try:
        if not os.path.exists(src):
            print(f"Le dossier src '{src}' n'existe pas.")
            return
        
        if os.path.exists(dst):
            print(f"Le dossier dst '{dst}' existe déjà.")
            shutil.rmtree(dst)
            print(f"Suppresion de {dst}.")
            shutil.copytree(src, dst)
            print(f"Le dossier '{src}' a été copié vers '{dst}'.")
        else:
            shutil.copytree(src, dst)
            print(f"Le dossier '{src}' a été copié vers '{dst}'.")
    except Exception as e:
        print(f"Une erreur s'est produite : {e}")

def create_dataset_test(classes_to_remove):
    """
    Create a test dataset.

    Args:
        classes_to_remove (string[]): classes to remove
    """
    dataset_dirname = "data/small-collomboles-n-classes/dataset"
    dataset_test_dirname = "data/small-collomboles-n-classes/dataset_test"

    copy_folder(dataset_dirname, dataset_test_dirname)

    min_class = 150
    dataset_test_dirname_train = os.path.join(dataset_test_dirname, "train")
    dataset_test_dirname_val = os.path.join(dataset_test_dirname, "val")
    classes_folder = os.listdir(dataset_test_dirname_train)
    classes_folder.sort()
    for class_folder in classes_folder:
        class_path = os.path.join(dataset_test_dirname_train, class_folder)
        images_path = os.listdir(class_path)
        n_image = len(images_path)
        print(f"{class_folder}: {n_image}")
        if n_image < min_class:
            min_class = n_image

    remove_classes(dataset_test_dirname_train, classes_to_remove)
    remove_classes(dataset_test_dirname_val, classes_to_remove)

    keep_only_n_images(dataset_test_dirname_train, n=10)
    keep_only_n_images(dataset_test_dirname_val, n=2)

if __name__ == "__main__":
    classes_to_remove = [
        "Allacma fusca",
        "Anurida maritima",
        "Bilobella aurantiaca",
        # "Bilobella braunerae",
        "Bourletiella arvalis",
        # "Bourletiella hortensis",
        "Brachystomella parvula",
        "Caprainea marginata",
        # "Ceratophysella denticulata",
        "Ceratophysella longispina",
        "Cyphoderus albinus",
        "Deuterosminthurus bicinctus",
        # "Deuterosminthurus pallipes",
        "Dicyrtoma fusca",
        "Dicyrtomina flavosignata",
        "Dicyrtomina minuta",
        # "Dicyrtomina ornata",
        "Dicyrtomina saundersi",
        "Dicyrtomina signata",
        "Entomobrya albocincta",
        "Entomobrya atrocincta",
        "Entomobrya corticalis",
        "Entomobrya marginata",
        "Entomobrya multifasciata",
        "Entomobrya muscorum",
        "Entomobrya nicoleti",
        "Entomobrya nigrocincta",
        "Entomobrya nivalis",
        # "Entomobrya superba",
        "Fasciosminthurus quinquefasciatus",
        "Folsomia candida",
        # "Folsomia quadrioculata",
        "Heteromurus major",
        # "Heteromurus nitidus",
        "Hypogastrura viatica",
        "Isotoma riparia",
        # "Isotoma viridis",
        "Isotomiella minor",
        "Isotomurus gallicus",
        "Isotomurus maculatus",
        "Isotomurus palustris",
        # "Isotomurus unifasciatus",
        "Kalaphorura burmeisteri",
        "Lepidocyrtus curvicollis",
        "Lepidocyrtus cyaneus",
        "Lepidocyrtus fimetarius",
        "Lepidocyrtus lignorum",
        # "Lepidocyrtus paradoxus",
        "Megalothorax minimus",
        "Monobella grassei",
        "Neanura muscorum",
        "Neelus murinus",
        "Orchesella bifasciata",
        "Orchesella cincta",
        "Orchesella flavescens",
        "Orchesella quinquefasciata",
        # "Orchesella villosa",
        "Parisotoma notabilis",
        "Podura aquatica",
        # "Pogonognathellus flavescens",
        "Pogonognathellus longicornis",
        # "Seira ferrarii",
        "Seira musarum",
        # "Sminthurides aquaticus",
        "Sminthurides malmgreni",
        "Sminthurides penicillifer",
        "Sminthurinus aureus",
        "Sminthurinus elegans",
        "Sminthurinus lawrencei",
        "Sminthurinus niger",
        "Sminthurinus trinotatus",
        "Sminthurus viridis",
        # "Tomocerus minor",
        "Tomocerus vulgaris",
        "Vertagopus asiaticus",
        "Vitronura giselae",
        # "Willowsia buski",
        "Willowsia nigromaculata",
        "Willowsia platani",
    ]
    create_dataset_test(classes_to_remove)

def classCounter(hierarchy_csv):
    with open(hierarchy_csv, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        NbClassesLevel = [0] * len(next(reader))
        KnownItems = []
        for row in reader:
            for i,col in enumerate(row):
                value = row[col]
                if value not in KnownItems:
                    NbClassesLevel[i] += 1
                    KnownItems.append(value)
    return NbClassesLevel
