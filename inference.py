#!/usr/bin/env python3
"""PyTorch Inference Script

An example inference script that outputs top-k class ids for images in a folder into a csv.

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""
import argparse
import json
import logging
import os
import time
import shutil
from contextlib import suppress
from functools import partial
from sys import maxsize

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import confusion_matrix

from timm.data import create_dataset, create_loader, resolve_data_config, ImageNetInfo, infer_imagenet_subset
from timm.layers import apply_test_time_pool
from timm.models import create_model
from timm.utils import AverageMeter, setup_default_logging, set_jit_fuser, ParseKwargs, accuracy

from scripts.metrics_logicseg import topk_accuracy_logicseg
from scripts.logic_seg_utils import *
from scripts.results import *
from scripts.metrics_hierarchy import *
from scripts.hierarchical_perfs_plot import *
from scripts.utils import *
from scripts.Logicseg_message_passing import *

try:
    from apex import amp
    has_apex = True
except ImportError:
    has_apex = False

try:
    from functorch.compile import memory_efficient_fusion
    has_functorch = True
except ImportError as e:
    has_functorch = False

has_compile = hasattr(torch, 'compile')


_FMT_EXT = {
    'json': '.json',
    'json-record': '.json',
    'json-split': '.json',
    'parquet': '.parquet',
    'csv': '.csv',
}

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('inference')


config_parser = parser = argparse.ArgumentParser(description='PyTorch ImageNet Inference')
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(description='PyTorch ImageNet Inference')

parser.add_argument('data', nargs='?', metavar='DIR', const=None,
                    help='path to dataset (*deprecated*, use --data-dir)')
parser.add_argument('--data-dir', metavar='DIR',
                    help='path to dataset (root dir)')
parser.add_argument('--dataset', metavar='NAME', default='',
                    help='dataset type + name ("<type>/<name>") (default: ImageFolder or ImageTar if empty)')
parser.add_argument('--split', metavar='NAME', default='validation',
                    help='dataset split (default: validation)')
parser.add_argument('--model', '-m', metavar='MODEL', default='resnet50',
                    help='model architecture (default: resnet50)')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--img-size', default=None, type=int,
                    metavar='N', help='Input image dimension, uses model default if empty')
parser.add_argument('--in-chans', type=int, default=None, metavar='N',
                    help='Image input channels (default: None => 3)')
parser.add_argument('--input-size', default=None, nargs=3, type=int, metavar='N',
                    help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
parser.add_argument('--use-train-size', action='store_true', default=False,
                    help='force use of train input size, even when test size is specified in pretrained cfg')
parser.add_argument('--crop-pct', default=None, type=float,
                    metavar='N', help='Input image center crop pct')
parser.add_argument('--crop-mode', default=None, type=str,
                    metavar='N', help='Input image crop mode (squash, border, center). Model default if None.')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float,  nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--num-classes', type=int, default=None,
                    help='Number classes in dataset')
parser.add_argument('--class-map', default='', type=str, metavar='FILENAME',
                    help='path to class to idx mapping file (default: "")')
parser.add_argument('--log-freq', default=10, type=int,
                    metavar='N', help='batch logging frequency (default: 10)')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--test-pool', dest='test_pool', action='store_true',
                    help='enable test time pool')
parser.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
parser.add_argument('--device', default='cuda', type=str,
                    help="Device (accelerator) to use.")
parser.add_argument('--amp', action='store_true', default=False,
                    help='use Native AMP for mixed precision training')
parser.add_argument('--amp-dtype', default='float16', type=str,
                    help='lower precision AMP dtype (default: float16)')
parser.add_argument('--model-dtype', default=None, type=str,
                   help='Model dtype override (non-AMP) (default: float32)')
parser.add_argument('--fuser', default='', type=str,
                    help="Select jit fuser. One of ('', 'te', 'old', 'nvfuser')")
parser.add_argument('--model-kwargs', nargs='*', default={}, action=ParseKwargs)
parser.add_argument('--torchcompile-mode', type=str, default=None,
                    help="torch.compile mode (default: None).")

parser.add_argument('--conf-matrix', action='store_true', default=False,
                    help="Make confusion matrix.")

# Custom parameter for LogicSeg
parser.add_argument('--logicseg', action='store_true', default=False,
                   help='Enable logicseg.')
parser.add_argument('--message-passing', action='store_true', default=False,
                   help='Apply logicseg message passing processing to output.')
parser.add_argument('--message-passing-iter-count', type=int, default=2,
                   help='number of iteration of the message passing.')
parser.add_argument('--csv-tree', default="", help="Path to hierarchy csv")

scripting_group = parser.add_mutually_exclusive_group()
scripting_group.add_argument('--torchscript', default=False, action='store_true',
                             help='torch.jit.script the full model')
scripting_group.add_argument('--torchcompile', nargs='?', type=str, default=None, const='inductor',
                             help="Enable compilation w/ specified backend (default: inductor).")
scripting_group.add_argument('--aot-autograd', default=False, action='store_true',
                             help="Enable AOT Autograd support.")

parser.add_argument('--results-dir', type=str, default=None,
                    help='folder for output results')
parser.add_argument('--results-file', type=str, default=None,
                    help='results filename (relative to results-dir)')
parser.add_argument('--results-format', type=str, nargs='+', default=['csv'],
                    help='results format (one of "csv", "json", "json-split", "parquet")')
parser.add_argument('--results-separate-col', action='store_true', default=False,
                    help='separate output columns per result index.')
parser.add_argument('--create-dir', action='store_true', default=True,
                    help='Create results-dir if the directory don\'t exist.')
parser.add_argument('--topk', default=1, type=int,
                    metavar='N', help='Top-k to output to CSV')
parser.add_argument('--fullname', action='store_true', default=False,
                    help='use full sample name in output (not just basename).')
parser.add_argument('--filename-col', type=str, default='filename',
                    help='name for filename / sample name column')
parser.add_argument('--index-col', type=str, default='index',
                    help='name for output indices column(s)')
parser.add_argument('--label-col', type=str, default='label',
                    help='name for output indices column(s)')
parser.add_argument('--output-col', type=str, default=None,
                    help='name for logit/probs output column(s)')
parser.add_argument('--output-type', type=str, default='prob',
                    help='output type colum ("prob" for probabilities, "logit" for raw logits)')
parser.add_argument('--label-type', type=str, default='description',
                    help='type of label to output, one of  "none", "name", "description", "detailed"')
parser.add_argument('--include-index', action='store_true', default=False,
                    help='include the class index in results')
parser.add_argument('--exclude-output', action='store_true', default=False,
                    help='exclude logits/probs from results, just indices. topk must be set !=0.')
parser.add_argument('--no-console-results', action='store_true', default=False,
                    help='disable printing the inference results to the console')



def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text

def main():
    setup_default_logging()
    args, _ = _parse_args()
    # might as well try to do something useful...
    args.pretrained = args.pretrained or not args.checkpoint

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    device = torch.device(args.device)

    # If logicSeg is used we create the class map file just before creating the dataset
    if args.logicseg:
        # Create the class map folder if it does not exist
        if not os.path.exists(args.class_map):
            os.makedirs(args.class_map)
        # generate the dictionary storing classes names as keys and assoiacted labels as values
        create_class_to_labels(args.csv_tree, args.class_map, verbose=False)

    model_dtype = None
    if args.model_dtype:
        assert args.model_dtype in ('float32', 'float16', 'bfloat16')
        model_dtype = getattr(torch, args.model_dtype)

    # resolve AMP arguments based on PyTorch / Apex availability
    amp_autocast = suppress
    if args.amp:
        assert model_dtype is None or model_dtype == torch.float32, 'float32 model dtype must be used with AMP'
        assert args.amp_dtype in ('float16', 'bfloat16')
        amp_dtype = torch.bfloat16 if args.amp_dtype == 'bfloat16' else torch.float16
        amp_autocast = partial(torch.autocast, device_type=device.type, dtype=amp_dtype)
        _logger.info('Running inference in mixed precision with native PyTorch AMP.')
    else:
        _logger.info('Running inference in float32. AMP not enabled.')

    if args.fuser:
        set_jit_fuser(args.fuser)

    #getting the path to the hierarchy.csv, to give it as a kwarg for our hierarchical classification model
    if args.csv_tree:
        if not hasattr(args, "model_kwargs"):
            args.model_kwargs = {}
        args.model_kwargs["hierarchy"] = args.csv_tree

    # create model
    in_chans = 3
    if args.in_chans is not None:
        in_chans = args.in_chans
    elif args.input_size is not None:
        in_chans = args.input_size[0]
    model = create_model(
        args.model,
        num_classes=args.num_classes,
        in_chans=in_chans,
        pretrained=args.pretrained,
        checkpoint_path=args.checkpoint,
        **args.model_kwargs,
    )
    if args.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes

    _logger.info(
        f'Model {args.model} created, param count: {sum([m.numel() for m in model.parameters()])}')

    data_config = resolve_data_config(vars(args), model=model)
    test_time_pool = False
    if args.test_pool:
        model, test_time_pool = apply_test_time_pool(model, data_config)

    model = model.to(device=device, dtype=model_dtype)
    model.eval()
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    if args.torchscript:
        model = torch.jit.script(model)
    elif args.torchcompile:
        assert has_compile, 'A version of torch w/ torch.compile() is required for --compile, possibly a nightly.'
        torch._dynamo.reset()
        model = torch.compile(model, backend=args.torchcompile, mode=args.torchcompile_mode)
    elif args.aot_autograd:
        assert has_functorch, "functorch is needed for --aot-autograd"
        model = memory_efficient_fusion(model)

    if args.num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu)))

    root_dir = args.data_dir # args.data or args.data_dir
    dataset = create_dataset(
        root=root_dir,
        name=args.dataset,
        split=args.split,
        class_map=args.class_map,
    )

    if test_time_pool:
        data_config['crop_pct'] = 1.0

    workers = 1 if 'tfds' in args.dataset or 'wds' in args.dataset else args.workers
    loader = create_loader(
        dataset,
        batch_size=args.batch_size,
        use_prefetcher=True,
        num_workers=workers,
        device=device,
        img_dtype=model_dtype or torch.float32,
        **data_config,
    )
    to_label = None
    if args.label_type in ('name', 'description', 'detail', 'especes'):
        imagenet_subset = infer_imagenet_subset(model)
        if imagenet_subset is not None :
            dataset_info = ImageNetInfo(imagenet_subset)
            if args.label_type == 'name':
                to_label = lambda x: dataset_info.index_to_label_name(x)
            elif args.label_type == 'detail':
                to_label = lambda x: dataset_info.index_to_description(x, detailed=True)
            # elif args.label_type == 'especes':
                # to_label = lambda x, class_to_label: get_label_branches(x, class_to_label)
            else:
                to_label = lambda x: dataset_info.index_to_description(x)
            to_label = np.vectorize(to_label)
        else:
            _logger.error("Cannot deduce ImageNet subset from model, no labelling will be performed.")

    top_k = min(args.topk, args.num_classes)
    batch_time = AverageMeter()
    end = time.time()
    all_indices = []
    all_labels = []
    all_outputs = []
    cm_all_ids_preds = []
    cm_all_labels_targets = []
    cm_all_targets = []
    h = 0
    labels_par_hauteur = None
    use_probs = args.output_type == 'prob'
    metrics_hierarchy = None
    nb_batches = 0

    with torch.no_grad():

        if args.csv_tree:
            '''Ces variables servent aux calcul des metriques pour tout les modèles'''
            H_raw, P_raw, M_raw = get_tree_matrices(args.csv_tree, verbose=False)
            La_raw = get_layer_matrix(args.csv_tree, verbose=False)
            metrics_hierarchy = MetricsHierarchy(H_raw, device)

            if args.logicseg:

                if args.message_passing:
                        # If message passing is enabled, create a new instance of the message passing class, initialized with the necessary matrices
                        message_passing = MessagePassing(H_raw, P_raw, M_raw, La_raw, args.message_passing_iter_count, device)
                

                # compute the label matrix
                label_matrix, _, index_to_node = get_label_matrix(args.csv_tree)
                # compute the labels associated with each class
                class_to_label = get_class_to_label(label_matrix, index_to_node)
                classes_labels = np.array(list(class_to_label.keys()))

                # useful data for the confusion matrix for each hierarchical level
                La = torch.tensor(La_raw).to(device) # (height, nb_nodes); La[i,j] = 1 if index node j is of depth i, otherwise 0
                h = La.shape[0] # height of the tree
                labels_par_hauteur = [[index_to_node[j] for j in range(La.shape[1]) if int(La[hauteur,j].item()) == 1 ] for hauteur in range(h)] # liste de h sous-listes; labels_par_hauteur[i] = les labels de la hauteur
                # "cm" = confusion_matrix
                # initialize the arrays storing the predicted indexes and the associated targets for each hierarchical level
                cm_par_hauteur_ids_preds = np.empty((h,0), dtype=np.float32)
                cm_par_hauteur_ids_targets = np.empty((h,0), dtype=np.float32)


        for batch_idx, (input, target) in enumerate(loader):
            nb_batches += 1
            with amp_autocast():
                output = model(input)

            if use_probs:
                output = output.softmax(-1)

            if args.conf_matrix and not args.logicseg:
                _, ids_preds = torch.max(output, 1)
                cm_all_ids_preds.append(ids_preds.cpu().numpy())
                cm_all_targets.append(target.cpu().numpy())

            if args.logicseg:
                if args.message_passing:
                    # If message passing is enabled, modify the model's output with the already initialized instance of the message passing class
                    output = message_passing.process(output)
                # apply the sigmoid
                output = torch.sigmoid(output)
                # calculer la probabilité associée à chaque branche
                logicseg_predictions = get_logicseg_predictions(output, label_matrix, device) # (nb_pred, nb_feuilles) probabilités des feuilles prédites par le modèle
                # construire le label associé à chaque branche
                onehot_targets = get_logicseg_predictions(target, label_matrix, device) # (nb_pred, nb_feuilles) one hot encoding des feuilles cibles
                # calculer les métriques sur les prédictions réalisées dans le batch courant
                metrics_hierarchy.compute_all_metrics(logicseg_predictions, onehot_targets, output, La)
                
                if args.conf_matrix:
                    # données utiles pour la matrice de confusion sur les feuilles
                    _, id_branch_output = logicseg_predictions.topk(1, dim=1) # proba max, id proba max: classe prédite par le modèles
                    _, id_branch_target = onehot_targets.topk(1, dim=1) # idem mais classe cible

                    predicted_labels = [classes_labels[id_branch_output[i]] for i in range(id_branch_output.shape[0])] # (nbre_pred, 1) stockant 1 chaine de caractères par ligne
                    target_labels = [classes_labels[id_branch_target[i]] for i in range(id_branch_target.shape[0])] # (nbre_pred, 1) stockant 1 chaine de caractères par ligne

                    cm_all_ids_preds.append(id_branch_output.cpu().numpy())
                    all_labels.append(predicted_labels)
                    cm_all_labels_targets.append(target_labels)
                    cm_all_targets.append(id_branch_target.cpu().numpy())

                    nb_pred = output.shape[0]

                    # extraction des probabilités de chaque noeud pour chaque hauteur de l'arbre:

                    output_rep = output.unsqueeze(0).repeat(h, 1, 1) # (h, nb_pred, nb_noeuds)
                    onehot_rep = target.unsqueeze(0).repeat(h, 1, 1) # (h, nb_pred, nb_noeuds)
                    La_rep = La.unsqueeze(1).repeat(1, nb_pred, 1) # (h, nb_pred, nb_noeuds)
                    probas_par_hauteur = output_rep * La_rep # (h, nb_pred, nb_noeuds): probas_par_hauteur[i,j,:] = les probas des noeuds de hauteur i pour la prédiction d'indice j
                    onehot_par_hauteur = onehot_rep * La_rep # (h, nb_pred, nb_noeuds)
                    _, id_branch_output = probas_par_hauteur.topk(1, dim=2) # (h, nb_pred) id de la classe prédite à chaque hauteur de l'arbre
                    _, id_branch_target = onehot_par_hauteur.topk(1, dim=2) # (h, nb_pred) id de la classe cible à chaque hauteur de l'arbre
                    
                    cm_par_hauteur_ids_preds = np.concatenate((cm_par_hauteur_ids_preds, id_branch_output.squeeze(2).cpu().numpy()), 1) # (h, nb_images_traitées)
                    cm_par_hauteur_ids_targets = np.concatenate((cm_par_hauteur_ids_targets, id_branch_target.squeeze(2).cpu().numpy()), 1) # (h, nb_images_traitées)
            
            elif args.csv_tree:
                '''Compute metrics for non-logicseg models'''
                #utiliser les node_to_index de get_label_matrix avec les hierarchy_lines
                classes = load_classnames(args.class_map)
                _, node_to_index, _ = get_label_matrix(args.csv_tree)

                new_targets = format_target(target, args.num_classes).T

                if args.multitask == True: #If model output contains predictions for several hierarchic levels
                    output = output[:,- args.num_classes :]

                augmented_output = add_nodes_to_output(args.csv_tree, output, classes, node_to_index).T
                augmented_targets = add_nodes_to_output(args.csv_tree, new_targets, classes, node_to_index).T

                metrics_hierarchy.compute_all_metrics(output, new_targets.to(device), augmented_output.to(device), La_raw, augmented_targets)
                print(metrics_hierarchy.get_metrics_string())

            ##

            if top_k and not args.logicseg:
                output, indices = output.topk(top_k)
                np_indices = indices.cpu().numpy()
                if args.include_index:
                    all_indices.append(np_indices)
                if to_label is not None:
                    np_labels = to_label(np_indices)
                    all_labels.append(np_labels)

            all_outputs.append(output.float().cpu().numpy())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % args.log_freq == 0:
                _logger.info('Predict: [{0}/{1}] Time {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
                    batch_idx, len(loader), batch_time=batch_time))

    all_indices = np.concatenate(all_indices, axis=0) if all_indices else None
    all_labels = np.concatenate(all_labels, axis=0) if all_labels else None
    all_outputs = np.concatenate(all_outputs, axis=0).astype(np.float32)
    filenames = loader.dataset.filenames(basename=not args.fullname)

    if args.create_dir:
        if not os.path.exists(args.results_dir):
            os.makedirs(args.results_dir)

    if args.conf_matrix:
        args_filepath = os.path.join(os.path.dirname(args.checkpoint), "args.yaml")
        copy_filepath = os.path.join(args.results_dir, "args.yaml")
        shutil.copy(args_filepath, copy_filepath)
        if args.logicseg:
            header_list = get_csv_header(args.csv_tree)
            # construire la matrice de confusion des feuilles
            cm_all_ids_preds = np.concatenate(cm_all_ids_preds, axis=0)
            cm_all_targets = np.concatenate(cm_all_targets, axis=0)
            # print("cm_all_targets: ", cm_all_targets)
            # print("cm_all_ids_preds: ", cm_all_ids_preds)
            cm = confusion_matrix(cm_all_targets, cm_all_ids_preds)
            cm_normalized = confusion_matrix(cm_all_targets, cm_all_ids_preds, normalize='true')
            np.savetxt(os.path.join(args.results_dir, "cm_branch.out"), cm)
            np.savetxt(os.path.join(args.results_dir, "cm_norm_branch.out"), cm_normalized)
            # construire la matrice de confusion pour chaque hauteur de l'arbre
            for hauteur in range(h):
                cm = confusion_matrix(cm_par_hauteur_ids_targets[hauteur,:], cm_par_hauteur_ids_preds[hauteur, :])
                cm_normalized = confusion_matrix(cm_par_hauteur_ids_targets[hauteur,:], cm_par_hauteur_ids_preds[hauteur, :], normalize='true')
                np.savetxt(os.path.join(args.results_dir, "cm_"+header_list[hauteur]+".out"), cm)
                np.savetxt(os.path.join(args.results_dir, "cm_norm_"+header_list[hauteur]+".out"), cm_normalized)
        else:
            cm_all_ids_preds = np.concatenate(cm_all_ids_preds, axis=0)
            cm_all_targets = np.concatenate(cm_all_targets, axis=0)
            cm = confusion_matrix(cm_all_targets, cm_all_ids_preds)
            cm_normalized = confusion_matrix(cm_all_targets, cm_all_ids_preds, normalize='true')
            np.savetxt(os.path.join(args.results_dir, "confusion_matrix.out"), cm)
            np.savetxt(os.path.join(args.results_dir, "confusion_matrix_norm.out"), cm_normalized)
            if not args.csv_tree and args.class_map:
                classes = load_classnames(args.class_map)
                save_metrics(cm, args.results_dir, f"metrics.csv", classes, "output")


    output_col = args.output_col or ('prob' if use_probs else 'logit')
    data_dict = {args.filename_col: filenames}
    if args.results_separate_col and all_outputs.shape[-1] > 1:
        if all_indices is not None:
            for i in range(all_indices.shape[-1]):
                data_dict[f'{args.index_col}_{i}'] = all_indices[:, i]
        if all_labels is not None:
            for i in range(all_labels.shape[-1]):
                data_dict[f'{args.label_col}_{i}'] = all_labels[:, i]
        for i in range(all_outputs.shape[-1]):
            data_dict[f'{output_col}_{i}'] = all_outputs[:, i]
    else:
        if all_indices is not None:
            if all_indices.shape[-1] == 1:
                all_indices = all_indices.squeeze(-1)
            data_dict[args.index_col] = list(all_indices)
        if all_labels is not None:
            if all_labels.shape[-1] == 1:
                all_labels = all_labels.squeeze(-1)
            data_dict[args.label_col] = list(all_labels)
        if all_outputs.shape[-1] == 1:
            all_outputs = all_outputs.squeeze(-1)
        data_dict[output_col] = list(all_outputs)

    df = pd.DataFrame(data=data_dict)

    results_filename = args.results_file
    if results_filename:
        filename_no_ext, ext = os.path.splitext(results_filename)
        if ext and ext in _FMT_EXT.values():
            # if filename provided with one of expected ext,
            # remove it as it will be added back
            results_filename = filename_no_ext
    else:
        # base default filename on model name + img-size
        img_size = data_config["input_size"][1]
        results_filename = f'{args.model}-{img_size}'

    if args.results_dir:
        results_filename = os.path.join(args.results_dir, results_filename)

    for fmt in args.results_format:
        save_results(df, results_filename, fmt)
    if args.conf_matrix and args.csv_tree:
        metrics_hierarchy.save_metrics_csv(os.path.join(args.results_dir, "metrics_results.csv"))
        if args.logicseg:
            print(f'--result')
            # Create the metrics file and store the data 
            # metrics_hierarchy.save_metrics_csv(os.path.join(args.results_dir, "metrics_results.csv"))
            with open(os.path.join(args.results_dir, "metrics_results.txt"), "w") as fichier:
                metrics_str = metrics_hierarchy.get_metrics_string()
                fichier.write(metrics_str)
                print(metrics_str)
    
            # Process and save confusion matrix of the leafs (in .out and .jpg files) /!\ names are important here
            cm = load_confusion_matrix(os.path.join(args.results_dir, "cm_branch.out"))
            cm_normalized = load_confusion_matrix(os.path.join(args.results_dir, "cm_norm_branch.out"))
            output_filename = "cm_branches.jpg"
            output_filename_norm = "cm_norm_branches.jpg"
            save_confusion_matrix(cm, output_filename, classes_labels, folder=args.results_dir)
            save_confusion_matrix(cm_normalized, output_filename_norm, classes_labels, folder=args.results_dir)
            df = save_metrics(cm, folder=args.results_dir, filename="metrics_branches.csv", classes=classes_labels, hierarchy_name="branches")

            classes = classes_labels
            hierarchy_lines = read_csv(args.csv_tree)
            hierarchy_lines_without_names = hierarchy_lines[1:]
            parents = get_parents(hierarchy_lines_without_names)
            hierarchy_names = hierarchy_lines[0]
            hierarchy_names.reverse()
            if not os.path.isdir(os.path.join(args.results_dir, "results_from_leaves")):
                os.mkdir(os.path.join(args.results_dir, "results_from_leaves"))
            save_confusion_matrix_and_metrics(os.path.join(args.results_dir, "results_from_leaves"), os.path.join(args.results_dir, "cm_branch.out"), classes, parents, hierarchy_names)

            header_list = get_csv_header(args.csv_tree)
            # Process and save confusion matrices of every tree levels (in .out and .jpg files) /!\ names are important here
            for hauteur in range(h):
                cm = load_confusion_matrix(os.path.join(args.results_dir, "cm_"+header_list[hauteur]+".out"))
                cm_normalized = load_confusion_matrix(os.path.join(args.results_dir, "cm_norm_"+header_list[hauteur]+".out"))
                if hauteur == 0:
                    cm = np.expand_dims(cm, (0,1)) # shape (1,1)
                    cm_normalized = np.expand_dims(cm_normalized, (0,1)) # shape (1,1)
                output_filename = "cm_im_"+header_list[hauteur]+".jpg"
                output_filename_norm = "cm_im_norm_"+header_list[hauteur]+".jpg"
                save_confusion_matrix(cm_normalized, output_filename, labels_par_hauteur[hauteur], folder=args.results_dir)
                save_confusion_matrix(cm_normalized, output_filename_norm, labels_par_hauteur[hauteur], folder=args.results_dir)
                next_df = save_metrics(cm, folder=args.results_dir, filename=f"metrics_{header_list[hauteur]}.csv", classes=labels_par_hauteur[hauteur], hierarchy_name=header_list[hauteur])
                df = pd.concat([df, next_df])
            
            df.to_csv(os.path.join(args.results_dir, "metrics_all.csv"), index=False)

            path_output_csv = os.path.join(args.results_dir, "results_from_leaves", "metric_F1_perfs.csv")
            # build the right csv file from metrics_all.csv without the lines whose "Etage" is "branches"
            # Attention il faut que le csv contienne les données calculées sur des matrices de confusion non normalisées
            build_F1_perfs_csv(os.path.join(args.results_dir, "results_from_leaves", "metrics_all.csv"), path_output_csv, args.csv_tree)
                # call the function
            color_list = get_custom_color_list(saturation_factor=1.25)
            plot_hierarchical_perfs(perfs_csv=path_output_csv,
                                        metric_to_plot="F1-score",
                                        cmap_list=color_list,
                                        show=False,
                                        html_output=os.path.join(args.results_dir, "F1_perfs.html"),
                                        png_output=os.path.join(args.results_dir, "F1_perfs.png"),
                                        remove_lines=False,
                                        font_size=32)

        else:

            '''TODO: Possiblement redondant'''
            classes = load_classnames(args.class_map)

            hierarchy_lines = read_csv(args.csv_tree)
            hierarchy_lines_without_names = hierarchy_lines[1:]
            parents = get_parents(hierarchy_lines_without_names)
            hierarchy_names = hierarchy_lines[0]
            hierarchy_names.reverse()
            save_confusion_matrix_and_metrics(args.results_dir, os.path.join(args.results_dir, "confusion_matrix.out"), classes, parents, hierarchy_names)

        # build the circle figure showing the F1 score for each node
        path_output_csv = os.path.join(args.results_dir, "metric_F1_perfs.csv")
            # build the right csv file from metrics_all.csv without the lines whose "Etage" is "branches"
            # TODO: ajouter la racine au csv ? (elle y est dans le csv de Edgar)
            # Attention il faut que le csv contienne les données calculées sur des matrices de confusion non normalisées
        build_F1_perfs_csv(os.path.join(args.results_dir, "metrics_all.csv"), path_output_csv, args.csv_tree)
            # call the function
        color_list = get_custom_color_list(saturation_factor=1.25)
        plot_hierarchical_perfs(perfs_csv=path_output_csv,
                                    metric_to_plot="F1-score",
                                    cmap_list=color_list,
                                    show=False,
                                    html_output=os.path.join(args.results_dir, "F1_perfs.html"),
                                    png_output=os.path.join(args.results_dir, "F1_perfs.png"),
                                    remove_lines=False,
                                    font_size=32)

def save_results(df, results_filename, results_format='csv', filename_col='filename'):
    np.set_printoptions(threshold=maxsize)
    results_filename += _FMT_EXT[results_format]
    if results_format == 'parquet':
        df.set_index(filename_col).to_parquet(results_filename)
    elif results_format == 'json':
        df.set_index(filename_col).to_json(results_filename, indent=4, orient='index')
    elif results_format == 'json-records':
        df.to_json(results_filename, lines=True, orient='records')
    elif results_format == 'json-split':
        df.to_json(results_filename, indent=4, orient='split', index=False)
    else:
        df.to_csv(results_filename, index=False)


if __name__ == '__main__':
    main()