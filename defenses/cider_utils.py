import types
from transformers import AutoModelForPreTraining, AutoTokenizer, AutoImageProcessor
import abc
import transformers
import json
import argparse
import os
import re
from PIL import Image
from tqdm import tqdm
import time
import base64
from pathlib import Path
from torchvision import transforms
import random
import io
import numpy as np
import pandas as pd
from math import ceil
import torch

try:
    from .cider_models.diffusion_denoiser.imagenet.DRM import DiffusionRobustModel
except ImportError:
    from defenses.cider_models.diffusion_denoiser.imagenet.DRM import DiffusionRobustModel


class QApair:
    """
    containing all informations for a query-answer process, including:
    - query
    - image path
    - refuse: bool, if is True(harmful image), VLM should directly refuse to answer.
    - answer
    - behaviour: label in Harmbench, will be used in scoring process.
    """

    def __init__(self, query, imgpth, refuse, ans, behav) -> None:
        self.query = query
        self.imgpth = imgpth
        self.refuse = refuse
        self.ans = ans
        self.behav = behav


class Defender:
    """
    contains methods:
        1. train: calculate threshold with given clean image data and specified ratio.
        2. predict: given img-text pair, return whether it is adversarial.
        3. get_confusion_matrix: given validation dataset, call predict function and return statistics (i.e. confusion matrix)
    """

    def __init__(self, threshold=None):
        self.threshold = threshold
        print(f"Defender initialized with threshold={self.threshold}")

    def train(self, data, ratio=0.9):
        """
        :data: cosine similarity between malicious query(text) & clean image
            cols: denoise times
            rows: different text
        :ratio: the ratio of clean image cosine similarity decrease value that should be lower than threshold
        """
        if type(data) == list:
            for df in data:
                names = [i for i in range(df.shape[1])]
                df.columns = names
            data = pd.concat(data, axis=0, ignore_index=True)
        base = data.iloc[:, 0]
        decrease = data.iloc[:, 1:].sub(base, axis=0)
        self.threshold = np.percentile(decrease, (1 - ratio) * 100)
        print(f"Threshold updated to {self.threshold}")

    def predict(self, cossims):
        """
        :cossims: a nD array of cosine similarity (m*n)
            cols: denoise times n
            row: m text
        return: a nD array of boolean (m*1)
            True means adversarial
        """
        if len(cossims.shape) == 1:
            return True in (np.array(cossims[1:]) - cossims[0] < self.threshold)
        else:
            ret = []
            for r in range(cossims.shape[0]):
                row = cossims.iloc[r]
                ret.append(
                    True in (np.array(row.iloc[1:]) - row.iloc[0] < self.threshold)
                )
            return ret

    def get_lowest_idx(self, cossims):
        """
        :cossims: a nD array of cosine similarity (1*n)
            cols: denoise times n
            row: 1 text
        return: a nD array of int (1*1)
            0 is clean, positive int is the index of lowest cosine similarity
        """
        # reshape the data
        cossims = np.array(cossims).reshape(1, -1)
        delta = cossims - cossims[0, 0]
        if np.min(delta) < self.threshold:
            return np.argmin(delta)
        else:
            return 0

    def get_confusion_matrix(self, datapath: str, checkpt_num=8, group=True):
        df = pd.read_csv(datapath)
        try:
            df = df[df["is_malicious"] == 1]  # only consider malicious text input
        except:
            pass
        results = {
            "constraint": [],
            "accuracy": [],
            "recall(dsr)": [],
            "precision": [],
            "f1": [],
            "classification threshold": [],
            "fpr": [],
        }
        fp, tn = 0, 0
        all_clean_header = [col for col in df.columns if "clean_" in col]
        clean_classes_names = set(
            ["_".join(h.split("_")[:2]) for h in all_clean_header]
        )
        for clean_class in clean_classes_names:
            clean_header = [col for col in df.columns if clean_class + "_" in col]
            if len(clean_header) > checkpt_num:
                clean_header = clean_header[:checkpt_num]
            clean_data = df[clean_header]
            clean_predict = np.array(self.predict(clean_data))
            fp += sum(clean_predict[:])
            tn += sum(~clean_predict[:])

        tot_tp, tot_fn = 0, 0
        all_adv_header = [col for col in df.columns if "prompt_" in col]
        adv_classes_names = set(["_".join(h.split("_")[:3]) for h in all_adv_header])
        for adv_class in adv_classes_names:
            adv_header = [col for col in df.columns if adv_class + "_" in col]
            if len(adv_header) > checkpt_num:
                adv_header = adv_header[:checkpt_num]
            adv_data = df[adv_header]
            adv_predict = np.array(self.predict(adv_data))
            tp = sum(adv_predict[:])
            fn = sum(~adv_predict[:])
            tot_tp += tp
            tot_fn += fn

            if not group:
                continue
            assert tp + fn == fp + tn

            acc = (tp + tn) / (tp + fn + fp + tn)
            recall = tp / (tp + fn)
            precision = tp / (tp + fp)
            f1 = 2 * precision * recall / (precision + recall)
            fpr = fp / (fp + tn)
            results["constraint"].append(adv_class.split("_")[-1])
            results["accuracy"].append(acc)
            results["recall(dsr)"].append(recall)
            results["precision"].append(precision)
            results["f1"].append(f1)
            results["classification threshold"].append(self.threshold)
            results["fpr"].append(fpr)
        if not group:
            tp = tot_tp
            fn = tot_fn
            acc = (tp + tn) / (tp + fn + fp + tn)
            recall = tp / (tp + fn)
            precision = tp / (tp + fp)
            f1 = 2 * precision * recall / (precision + recall)
            fpr = fp / (fp + tn)
            results["constraint"].append("all")
            results["accuracy"].append(acc)
            results["recall(dsr)"].append(recall)
            results["precision"].append(precision)
            results["f1"].append(f1)
            results["classification threshold"].append(self.threshold)
            results["fpr"].append(fpr)
        results = pd.DataFrame(results)
        return results


class Encoder:
    model_path = None

    def __init__(self, mdpth) -> types.NoneType:
        self.model_path = mdpth

    @staticmethod
    def compute_cosine(a_vec: np.ndarray, b_vec: np.ndarray):
        """calculate cosine similarity"""
        norms1 = np.linalg.norm(a_vec, axis=1)
        norms2 = np.linalg.norm(b_vec, axis=1)
        dot_products = np.sum(a_vec * b_vec, axis=1)
        cos_similarities = dot_products / (norms1 * norms2)
        return cos_similarities[0]

    @abc.abstractmethod
    def calc_cossim(self, pairs: list[tuple[str, str]]):
        res = []
        for p in pairs:
            text_embed = self.embed_text(p[0])
            img_embed = self.embed_img(p[1])
            cossim = self.compute_cosine(text_embed, img_embed)
            res.append(cossim)
        return res

    @abc.abstractmethod
    def embed_img(self, imgpth) -> np.ndarray:
        pass

    @abc.abstractmethod
    def embed_text(self, text) -> np.ndarray:
        pass


class LlavaEncoder(Encoder):
    def __init__(self, mdpth, device="cuda:0") -> types.NoneType:
        super().__init__(mdpth)
        self.device = device
        # loaded through multimodalmodels.loaders (pinned revision for registered ids)
        from multimodalmodels.loaders import load_pretraining_encoder
        self.model, self.tokenizer, self.imgprocessor = load_pretraining_encoder(
            mdpth, torch_dtype="float16", low_cpu_mem_usage=True, to_device=device
        )

    def embed_img(self, imgpth) -> np.ndarray:
        image = Image.open(imgpth)
        pixel_value = self.imgprocessor(image, return_tensors="pt").pixel_values.to(
            self.device
        )
        image_outputs = self.model.vision_tower(pixel_value, output_hidden_states=True)
        selected_image_feature = image_outputs.hidden_states[
            self.model.config.vision_feature_layer
        ]
        selected_image_feature = selected_image_feature[:, 1:]  # by default
        image_features = self.model.multi_modal_projector(selected_image_feature)
        image_features = torch.mean(image_features, dim=1).detach().to("cpu").numpy()
        return image_features

    def embed_text(self, text) -> np.ndarray:
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        input_embeds = self.model.get_input_embeddings()(input_ids)
        input_embeds = torch.mean(input_embeds, dim=1).detach().to("cpu").numpy()
        return input_embeds


def generate_denoised_img(model="diffusion", **kwargs):
    import shutil

    try:
        shutil.rmtree(kwargs["save_dir"])
    except FileNotFoundError:
        print("No existing temp dir")
    finally:
        os.makedirs(kwargs["save_dir"], exist_ok=True)

    if model == "diffusion":
        func = generate_denoised_img_diffusion
    # elif model == "dncnn":
    #     func = generate_denoised_img_DnCNN
    # elif model == "nlm":
    #     func = generate_denoised_img_NLM
    else:
        raise RuntimeError(f"Unrecognised model type: {model}")
    return func(**kwargs)


def generate_denoised_img_diffusion(
    imgpth: list[str],
    save_dir: str,
    cps: int,
    step=50,
    device: int = 0,
    batch_size=50,
    denoiser=None,
    model_path=None,
    **kwargs,
):

    resized_imgs = []
    denoised_pth = []
    imgpth.sort()

    trans = transforms.Compose([transforms.Resize([224, 224])])
    for filepath in imgpth:
        img = Image.open(filepath).convert("RGB")
        filename = os.path.split(filepath)[1]
        img = trans(img)
        resized_imgs.append((img, filename))
        savename = os.path.splitext(filename)[0] + "_denoised_000times.jpg"
        img.save(os.path.join(save_dir, savename))
        denoised_pth.append(os.path.join(save_dir, savename))
    if cps <= 1:
        return denoised_pth

    device_obj = f"cuda:{device}" if isinstance(device, int) else str(device)
    owns_model = denoiser is None
    model = denoiser or DiffusionRobustModel(
        device_obj=device_obj, model_path=model_path, load_classifier=False
    )
    iterations = range(step, cps * step, step)
    b_num = ceil(len(resized_imgs) / batch_size)
    for b in tqdm(range(b_num), desc="denoising batch"):
        l = b * batch_size
        r = (b + 1) * batch_size if b < b_num - 1 else len(resized_imgs)
        part = resized_imgs[l:r]
        for it in iterations:
            ary = [np.array(_[0], dtype=np.float32) / 255 * 2 - 1 for _ in part]
            ary = np.array(ary)
            ary = torch.tensor(ary).permute(0, 3, 1, 2).to(device_obj)
            denoised_ary = np.array(model.denoise(ary, it).to("cpu"))
            denoised_ary = denoised_ary.transpose(0, 2, 3, 1)
            denoised_ary = (denoised_ary + 1) / 2 * 255
            denoised_ary = denoised_ary.astype(np.uint8)
            for i in range(denoised_ary.shape[0]):
                img = Image.fromarray(denoised_ary[i])
                sn = os.path.splitext(part[i][1])[
                    0
                ] + "_denoised_{:0>3d}times.jpg".format(it)
                img.save(os.path.join(save_dir, sn))
                denoised_pth.append(os.path.join(save_dir, sn))
    if owns_model:
        del model
        torch.cuda.empty_cache()
    return denoised_pth


def get_similarity_list(
    text_file,
    pair_mode,
    encoder_pth,
    imgdir="./temp/denoised_imgs",
    device="cuda:0",
    cpnum=8,
):

    if "llava" in encoder_pth.lower():
        encoder = LlavaEncoder(encoder_pth, device)
    else:
        raise ValueError(f"Unrecognised encoder Type from:{encoder_pth}")

    # load text inputs
    # with open(text_file) as fr:
    #     reader = csv.reader(fr)
    #     queries = [line[0] for line in reader if line[1]=="standard"]

    # with open(text_file, 'r', encoding='utf-8') as f:
    #     attack_cases = json.load(f)
    #     queries = [case['attack_prompt'] for case in attack_cases]

    with open(text_file, "r", encoding="utf-8") as f:
        attack_cases = json.load(f)
        # Handle both list and dictionary formats
        if isinstance(attack_cases, dict):
            # Convert dictionary to list of values
            attack_cases = list(attack_cases.values())
        elif isinstance(attack_cases, list):
            attack_cases = attack_cases
        queries = [case.get("attack_prompt", "") for case in attack_cases]

    # print(f"Loading {len(queries)} queries from {text_file}")
    # print(queries)
    dir1 = sorted(os.listdir(imgdir))
    img_pths = [os.path.join(imgdir, img) for img in dir1]
    # compute cosine similarity between text and n denoised images
    # and form a table of size ((len(queries),image_num, ckpt_num)) or (len(text_embed_list), ckpt_num)
    image_num = len(img_pths) // cpnum
    if pair_mode == "combine":
        cossims = np.zeros((len(queries), image_num, cpnum))
        for i in range(len(queries)):
            for j in range(image_num):
                inputs = [
                    (queries[i], img_pths[k]) for k in range(j * cpnum, (j + 1) * cpnum)
                ]
                temp = encoder.calc_cossim(inputs)
                cossims[i, j] = temp
    else:
        cossims = np.zeros((len(queries), cpnum))
        for i in range(image_num):
            inputs = [
                (queries[i], img_pths[k]) for k in range(i * cpnum, (i + 1) * cpnum)
            ]
            temp = encoder.calc_cossim(inputs)
            cossims[i] = temp
    return cossims


def defence(
    imgpth: list[str], args, cpnum=8
) -> tuple[list[str] | list[list], list[bool] | list[list[bool]]]:
    image_num = len(imgpth)
    denoised_img_dir = os.path.join(args.tempdir, "denoised_imgs")

    if args.denoiser == "dncnn":
        cpnum = 2
    denoised_imgpth = generate_denoised_img(
        args.denoiser,
        imgpth=imgpth,
        save_dir=denoised_img_dir,
        cps=cpnum,
        batch_size=50,
        device=args.cuda,
        model_path=[args.denoiser_path],
    )
    denoised_imgpth.sort()

    sim_matrix = get_similarity_list(
        args.text_file,
        args.pair_mode,
        imgdir=denoised_img_dir,
        cpnum=cpnum,
        device=args.cuda,
        encoder_pth=args.encoder_path,
    )

    d_denoise = Defender(threshold=args.threshold)
    adv_idx = []
    refuse = []
    ret_images_pths = []
    if args.pair_mode == "combine":
        text_num = sim_matrix.shape[0]

        for i in range(text_num):
            adv_row, refuse_row, ret_img_row = [], [], []
            for j in range(image_num):
                low = d_denoise.get_lowest_idx(sim_matrix[i][j])
                adv_row.append(low)
                refuse_row.append(low != 0)
                idx = j * cpnum + low
                ret_img_row.append(denoised_imgpth[idx])
            adv_idx.append(adv_row)
            refuse.append(refuse_row)
            ret_images_pths.append(ret_img_row)
    else:
        for i in range(sim_matrix.shape[0]):
            adv_idx.append(d_denoise.get_lowest_idx(sim_matrix[i]))
            refuse.append(adv_idx[i] != 0)
            idx = i % image_num * cpnum + adv_idx[i]
            ret_images_pths.append(denoised_imgpth[idx])

    return (ret_images_pths, refuse)


_CIDER_DENOISERS = {}


def get_denoiser(device=0, model_path=None):
    device_obj = f"cuda:{device}" if isinstance(device, int) else str(device)
    key = (device_obj, os.path.abspath(model_path) if model_path else None)
    if key not in _CIDER_DENOISERS:
        _CIDER_DENOISERS[key] = DiffusionRobustModel(
            device_obj=device_obj,
            model_path=model_path,
            load_classifier=False,
        )
    return _CIDER_DENOISERS[key]


def cider_detect_single(image_path, prompt, encoder, denoiser, threshold,
                        tmp_dir, cpnum=8, step=50, device=0):
    import os
    denoised_dir = os.path.join(tmp_dir, "denoised_imgs")
    os.makedirs(denoised_dir, exist_ok=True)
    denoised_pths = generate_denoised_img_diffusion(
        imgpth=[image_path],
        save_dir=denoised_dir,
        cps=cpnum,
        step=step,
        device=device,
        batch_size=50,
        denoiser=denoiser,
    )
    denoised_pths.sort()

    pairs = [(prompt, p) for p in denoised_pths]
    cossims = encoder.calc_cossim(pairs)

    detector = Defender(threshold=threshold)
    low_idx = detector.get_lowest_idx(cossims)
    is_adversarial = (low_idx != 0)

    return is_adversarial, denoised_pths
