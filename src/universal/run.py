## Built-in libraries.
import sys
import os
import io
import time 

## Third party libraries.
import numpy as np
import tqdm
import torch 
import torch.nn.functional as func

from .parameters   import Parameters
from .builder      import Builder
from .adversarial  import UniversalAdversarialExample
from .perturbation import UniversalPerturbation
from .loss         import UniversalAdversarialLoss
from .loss         import UniversalL2Similarity, UniversalLocalSimilarity, UniversarlCrossL2Similarity
from .loss         import UniversalIsospec
from .loss         import ZeroLoss

def run(parameters: Parameters) -> UniversalAdversarialExample:
    start_time = time.time() 
  
    REPO_ROOT   = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath('__file__')),".."))
    DEVICE      = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    SRC_DIR     = os.path.join(REPO_ROOT, "src")
    DATASET     = os.path.join(REPO_ROOT, parameters.dataset)
    PARAMS_FILE = os.path.join(REPO_ROOT, parameters.dataset_file)

    ## Repository top level modules.
    sys.path.insert(0, SRC_DIR)
    import models
    import dataset
    from utils import visualize_and_compare

    ## Dataset
    if parameters.name == "smal":
        traindata = dataset.SmalDataset(DATASET, device=DEVICE, train=True, test=False, custom=False, transform_data=True)
        if parameters.datasubset == "train":
            dataset = dataset.SmalDataset(DATASET, device=DEVICE, train=False, test=True, transform_data=False)
        elif parameters.datasubset == "test":
            dataset = dataset.SmalDataset(DATASET, device=DEVICE, train=False, test=True, transform_data=False)
        elif parameters.datasubset == "custom":
            custom_list = parameters.datasubset_list
            dataset = dataset.SmalDataset(DATASET, device=DEVICE, train=False, test=False, custom=True, custom_list=custom_list, transform_data=False)
    else:
        assert()

    ## Classifier
    model = models.ChebnetClassifier(param_conv_layers = [128,128,64,64],
                                     D_t = traindata.downscale_matrices,
                                     E_t = traindata.downscaled_edges,
                                     num_classes = traindata.num_classes,
                                     parameters_file = PARAMS_FILE).to(DEVICE)
    print(model)

    ## Template index.
    template_index = parameters.template_index

    ## Targets list, using the second logit for each one in this case.
    # targets = [-1] * len(dataset)
    # for shape_index in range(len(dataset)):
    #       shape = dataset[shape_index]
    #       logits = model(shape.pos)
    #       _, index = torch.sort(logits, dim=0)
    #       targets[shape_index] = index[-2]

    targets = parameters.targets if parameters.targets else None

    ## Configure adversarial example components using builder.
    #--------------------------------------------------------
    builder = Builder(search_iterations=1)
    builder.set_classifier(model)
    builder.set_dataset(dataset)
    builder.set_target(targets)
    builder.set_template_index(template_index)

    ## Perturbation & Losses.
    # Original formulation (with alpha_i's).
    builder.set_perturbation(UniversalPerturbation)
    # Similarity Losses.
    builder.set_similarity_loss(ZeroLoss)
    builder.set_cross_similarity_loss(ZeroLoss)
    # Adversarial Loss.
    builder.set_adversarial_loss(UniversalAdversarialLoss)
    # Isospectralization loss.
    builder.set_isospectralization_loss(UniversalIsospec)
    # Regularization Loss
    builder.set_regularization_loss(ZeroLoss)

    adversarial_sample = builder.build(**parameters.builder_args())

    end_time = time.time() 
    print(f"Total runtime {end_time - start_time}") 

    return adversarial_sample

from utils       import off
from utils.tests import create_timestamped_folder
from .utils      import save, save_params, load_params, save_dic, save_list
import pprint

def stats(adversarial_sample: UniversalAdversarialExample, 
          params: Parameters,
          save_data: bool = False,
          save_off: bool = False,
          save_parameters: bool = False,
          verbose: bool = False):
    r"""Print some stats of the adversarial sample just obtained.
        Set the flags to save automatically the data, parameters and off 
        files for the perturbed shapes.
        Set verbose to print the logits for all shapes, otherwise 
        just prints the summary at the end.
     """

    if save_off or save_data or save_parameters:
        STR_FOLDER_TEMPLATE = REPO_ROOT + '/out/universal-'
        str_folder = create_timestamped_folder(STR_FOLDER_TEMPLATE)

    if save_data:
        STR_OUPUT_FILE = 'data.pt'
        save(adversarial_sample, os.path.join(str_folder, STR_OUPUT_FILE))

    if save_parameters:
        STR_PARAMS_FILE = "params.json"
        save_params(params, os.path.join(str_folder, STR_PARAMS_FILE))
        STR_LOSS_FILE = "loss.json"
        save_losses(adversarial_sample.logger, os.path.join(str_folder, STR_LOSS_FILE))

    # Print out results.
    stats_lbl_original = []
    stats_lbl_prediction = []
    stats_lbl_prediction_i = []

    stats_alpha_list = []
    stats_alpha_i_list = []

    for index in range(adversarial_sample.shape_count()):
      str_shape = adversarial_sample.file_name(index)
      if verbose: print("===", str_shape)

      model = lambda pos: adversarial_sample.classify(pos)

      ## Original classification.
      original_classification = model(adversarial_sample.pos(index))
      if verbose: print("Original shape logits:\n", original_classification)

      _, lbl_original = original_classification.max(dim=-1)

      stats_lbl_original.append(lbl_original.item())

      perturbed_pos_i = adversarial_sample.perturbed_positions_i(index)

      out: torch.Tensor = model(perturbed_pos_i)
      if verbose: print("--\nAlpha_i\nPerturbed shape:\n", out)

      _, lbl_prediction = out.max(dim=-1)

      if verbose:
        print("Successful" if not int(lbl_prediction) == int(lbl_original) else "Unsuccessful",
              "[prediction: ", int(lbl_prediction),
              ", original: ", int(lbl_original), "]")

      stats_alpha_i_list.append((str_shape, not int(lbl_prediction) == int(lbl_original)))
      stats_lbl_prediction_i.append(lbl_prediction.item())

      if (save_off):
        off.write_off(adversarial_sample.pos(index), adversarial_sample.faces(index), os.path.join(str_folder, str_shape + ".off"))
        if (adversarial_sample.perturbation.is_alpha_i()): 
            off.write_off(perturbed_pos_i, adversarial_sample.faces(index), os.path.join(str_folder, str_shape + "_perturbed_i.off"))

    # Print full summary.
    stroutput = io.StringIO()

    if (adversarial_sample.perturbation.is_alpha_i()): 
        print("alpha_i : ", file = stroutput)
        pprint.pprint(stats_alpha_i_list, stream = stroutput)

        alphais = list(map((lambda x: x[1]), stats_alpha_i_list))
        print(alphais, file = stroutput)
        print(all(alphais), file = stroutput)

    print("labels            : ", stats_lbl_original, file = stroutput)
    if (adversarial_sample.perturbation.is_alpha_i()): 
        print("labels prediciton : ", stats_lbl_prediction_i, file = stroutput)

    print(stroutput.getvalue())

    if save_data:
        STR_STATS_FILE = "stats.txt"
        filehandler = open(os.path.join(str_folder, STR_STATS_FILE), 'w+')
        filehandler.write(stroutput.getvalue())
        filehandler.close()
    stroutput.close()
    return


def save_results(adversarial_sample: UniversalAdversarialExample, 
          str_out_dir: str,
          params, 
          custom_list):
    r"""Save all.
     """

    STR_OUPUT_FILE = 'adv_ex.pt'
    save(adversarial_sample, os.path.join(str_out_dir, STR_OUPUT_FILE))

    STR_LOSS_FILE = "loss.json"
    save_dic(adversarial_sample.logger, os.path.join(str_out_dir, STR_LOSS_FILE))

    STR_PARAM_FILE = "params.json"
    save_dic(params, os.path.join(str_out_dir, STR_PARAM_FILE))

    STR_DATALIST_FILE = "custom_dataset_list.json"
    save_list(custom_list, os.path.join(str_out_dir, STR_DATALIST_FILE))

    all_lbl_original = []
    all_lbl_prediction = []
    success_list = []
    if adversarial_sample.target:
        target_list = []
    model = lambda pos: adversarial_sample.classify(pos)

    for index in range(adversarial_sample.shape_count()):
      str_shape = adversarial_sample.file_name(index)

      ## Original classification.
      original_classification = model(adversarial_sample.pos(index))
      _, lbl_original = original_classification.max(dim=-1)
      all_lbl_original.append(lbl_original.item())

      ## Targets.
      if adversarial_sample.target:
          target_list.append(adversarial_sample.target[index].item())

      ## Predicted classification.
      perturbed_pos_i = adversarial_sample.perturbed_positions_i(index)
      out: torch.Tensor = model(perturbed_pos_i)
      _, lbl_prediction = out.max(dim=-1)
      all_lbl_prediction.append(lbl_prediction.item())

      success = (lbl_prediction != lbl_original).item()
      success_list.append(success)

      off.write_off(adversarial_sample.pos(index), adversarial_sample.faces(index), os.path.join(str_out_dir, str_shape + ".off"))
      off.write_off(perturbed_pos_i, adversarial_sample.faces(index), os.path.join(str_out_dir, str_shape + "_perturbed.off"))

    success_rate = sum(success_list) / adversarial_sample.shape_count()

    STR_SUCCESS_FILE = "success_list.json"
    save_list(success_list, os.path.join(str_out_dir, STR_SUCCESS_FILE))

    # Print full summary.https://www.askpython.com/python/built-in-methods/python-print-to-file
    stroutput = io.StringIO()

    print("## Adversarial attack on", adversarial_sample.shape_count(), "shapes ##", file = stroutput)
    print("\n\nIndices of shapes of the testset used:\n ", custom_list, file = stroutput)
    print("\nlabels            : ", all_lbl_original, file = stroutput)
    print("labels prediction : ", all_lbl_prediction, file = stroutput)
    if adversarial_sample.target:
        print("labels target     : ", target_list, file = stroutput)
    print("\nSuccess rate      : ", success_rate, file = stroutput)
    print("\nFinal losses:", file = stroutput)
    print("Isospec     : %.3e" %adversarial_sample.logger["isospectralization"][-1], file = stroutput)
    print("Adversarial : %.3e" %adversarial_sample.logger["adversarial"][-1], file = stroutput)
    print("Similarity  : %.3e" %adversarial_sample.logger["similarity"][-1], file = stroutput)
    print('\n\nParameters dictionary:\n', params, file = stroutput)

    print(stroutput.getvalue())

    STR_STATS_FILE = "summary.txt"
    filehandler = open(os.path.join(str_out_dir, STR_STATS_FILE), 'w+')
    filehandler.write(stroutput.getvalue())
    filehandler.close()
    stroutput.close()
    return


def save_validation(pos, perturbed_pos, faces, 
          model,
          str_out_dir: str,
          params, 
          all_losses,
          logits,
          custom_list):
    r"""Save all for isospec on a validation set.
     """

    #STR_OUPUT_FILE = 'adv_ex.pt'
    #save(adversarial_sample, os.path.join(str_out_dir, STR_OUPUT_FILE))

    STR_LOSS_FILE = "loss.json"
    save_list(all_losses, os.path.join(str_out_dir, STR_LOSS_FILE))

    STR_LOGIT_FILE = "logits.json"
    save_list(logits, os.path.join(str_out_dir, STR_LOGIT_FILE))

    STR_PARAM_FILE = "params.json"
    save_dic(params, os.path.join(str_out_dir, STR_PARAM_FILE))

    STR_DATALIST_FILE = "new_shapes_list.json"
    save_list(custom_list, os.path.join(str_out_dir, STR_DATALIST_FILE))

    all_lbl_original = []
    all_lbl_prediction = []
    final_losses = []
    success_list = []
    cmodel = lambda x : model(x.float())

    for index in range(len(custom_list)):
      str_shape = 'shape_' + str(custom_list[index])

      ## Original classification.
      pos_i = pos[index]
      original_classification = cmodel(pos_i)
      _, lbl_original = original_classification.max(dim=-1)
      all_lbl_original.append(lbl_original.item())

      ## Predicted classification.
      perturbed_pos_i = perturbed_pos[index]
      out: torch.Tensor = cmodel(perturbed_pos_i)
      _, lbl_prediction = out.max(dim=-1)
      all_lbl_prediction.append(lbl_prediction.item())

      success = (lbl_prediction != lbl_original).item()
      success_list.append(success)
      final_losses.append(all_losses[index][-1])

      off.write_off(pos_i, faces, os.path.join(str_out_dir, str_shape + ".off"))
      off.write_off(perturbed_pos_i, faces, os.path.join(str_out_dir, str_shape + "_perturbed.off"))

    success_rate = sum(success_list) / len(custom_list)

    STR_SUCCESS_FILE = "success_list.json"
    save_list(success_list, os.path.join(str_out_dir, STR_SUCCESS_FILE))

    # Print full summary.
    stroutput = io.StringIO()

    print("## Adversarial attack on", len(custom_list), "unseen shapes ##", file = stroutput)
    print("\n\nIndices of shapes used for validation:\n ", custom_list, file = stroutput)
    print("\nlabels            : ", all_lbl_original, file = stroutput)
    print("labels prediction : ", all_lbl_prediction, file = stroutput)
    print("\nSuccess rate      : ", success_rate, file = stroutput)
    print("\nFinal losses      : ", final_losses, file = stroutput)
    #print("Isospec     : %.3e" %adversarial_sample.logger["isospectralization"][-1], file = stroutput)
    #print("Adversarial : %.3e" %adversarial_sample.logger["adversarial"][-1], file = stroutput)
    #print("Similarity  : %.3e" %adversarial_sample.logger["similarity"][-1], file = stroutput)
    print('\n\nParameters dictionary:\n', params, file = stroutput)

    print(stroutput.getvalue())

    STR_STATS_FILE = "summary.txt"
    filehandler = open(os.path.join(str_out_dir, STR_STATS_FILE), 'w+')
    filehandler.write(stroutput.getvalue())
    filehandler.close()
    stroutput.close()
    return

