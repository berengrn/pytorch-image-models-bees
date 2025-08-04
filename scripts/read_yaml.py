import yaml
import os

def read_yaml(filepath):
    """
    Read YAML file.

    Args:
        filepath (string): filepath to the yaml file
    
    Returns:
        dict{}: yaml data
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
            return data
    except FileNotFoundError:
        print("Fichier non trouvé :", filepath)
    except yaml.YAMLError as e:
        print("Erreur lors de la lecture du YAML :", e)

def compute_model_name(filepath):
    """
    Compute model name from args.yaml file.

    Args:
        filepath (string): filepath to the yaml file

    Returns:
        string: model name
        string: model name for filename
    """
    data = read_yaml(filepath)
    modified_logicseg = data["modified_logicseg"] if "modified_logicseg" in data else False
    logicseg = data["logicseg"]
    softlabel = data.get("softlabels", False)
    is_logicseg = modified_logicseg or logicseg
    if is_logicseg:
        # print(data)
        method = data["logicseg_method"]
        message_passing = data.get("message_passing", False)
        crule_loss_weight = data["crule_loss_weight"]
        if message_passing:
            if method == "bce":
                return fr"LogicSeg MP (BCE, $\alpha$={crule_loss_weight})", fr"logicseg_bce_mp{str(crule_loss_weight).replace('.', '_').replace(' ', '_')}"
            elif method == "multi_bce":
                return fr"LogicSeg MP (Multi BCE, $\alpha$={crule_loss_weight})", fr"logicseg_multibce_mp{str(crule_loss_weight).replace('.', '_').replace(' ', '_')}"
            else:
                return fr"LogicSeg MP ($\alpha$={crule_loss_weight})", fr"logicseg{str(crule_loss_weight).replace('.', '_').replace(' ', '_')}"
        else:
            if method == "bce":
                return fr"LogicSeg (BCE, $\alpha$={crule_loss_weight})", fr"logicseg_bce{str(crule_loss_weight).replace('.', '_').replace(' ', '_')}"
            elif method == "multi_bce":
                return fr"LogicSeg (Multi BCE, $\alpha$={crule_loss_weight})", fr"logicseg_multibce{str(crule_loss_weight).replace('.', '_').replace(' ', '_')}"
            else:
                return fr"LogicSeg ($\alpha$={crule_loss_weight})", fr"logicseg{str(crule_loss_weight).replace('.', '_').replace(' ', '_')}"
    elif softlabel:
        softlabels_beta = data["softlabels_beta"]
        return fr"Soft Label ($\beta$={softlabels_beta})", f'soft_label_{str(softlabels_beta).replace('.', '_').replace(' ', '_')}'
    else:
        hce = data["hce_loss"]
        bce = data["bce_loss"]
        hce_alpha = data["hce_alpha"]
        if hce:
            return fr"HCE ($\alpha$={hce_alpha})", f'hce_alpha_{str(hce_alpha).replace('.', '_').replace(' ', '_')}'
        elif bce:
            return "BCE", "bce"
        else:
            return "CCE", "cce"

def get_method(data):
    """
    Get the method use in logicseg.

    Args:
        data (dict{}): yaml data

    Returns:
        string: method
    """
    if "method" in data:
        return data["method"]
    else:
        modified_logicseg = data["modified_logicseg"] if "modified_logicseg" in data else False
        logicseg = data["logicseg"]
        if logicseg:
            return "bce"
        elif modified_logicseg:
            return "multi_bce"
        else:
            return "error"