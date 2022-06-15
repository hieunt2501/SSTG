import os
import time
import re
import torch
import pickle
import logging
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm
from collections import defaultdict

from torch import Tensor, nn
from torch.utils.data import DataLoader

from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AdamW
# from datasets import load_metric
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, recall_score, fbeta_score, precision_score

# rouge = load_metric("rouge")
# bleu = load_metric("bleu")


from constants import TAGS
from dataset_gen import InferCollate, InferDataset, MyCollate
from data_helper.word_tokenize import insert_dummy

logging.basicConfig(filename='train.log', level=logging.DEBUG)
logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self,
                 device,
                 model,
                 n_words,
                 bos_token_id=106490,
                 eos_token_id=106491,
                 resume_epoch=0,
                 change_weight=1.5,
                 append_weight=1.5,
                 nwords=64000,
                 max_src_len=512,
                 logging_steps=4000,
                 pad_id=0,
                 infer=False,
                 char_tokenizer=None,
                 tokenizer=None,
                 train_dataset=None,
                 val_dataset=None,
                 model_dir="",
                 resume=False,
                 checkpoint="",
                 max_clipnorm=1,
                 tag_size=4,
                 epochs=1,
                 batch_size=2,
                 alpha=3,
                 lr=1e-5):

        # model config
        self.device = device
        self.model = model.to(self.device)
        self.batch_size = batch_size
        self.epochs = epochs
        self.resume_epoch = resume_epoch
        # model param
        self.lr = lr
        self.max_clipnorm = max_clipnorm
        self.tag_size = tag_size
        self.change_weight = change_weight
        self.n_words = n_words

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.tokenizer = tokenizer
        self.char_tokenizer = char_tokenizer
        self.checkpoint = checkpoint
        self.logging_steps = logging_steps
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id
        # data config
        self.pad_id = pad_id
        self.max_src_len = max_src_len
        self.change_weight = change_weight
        self.append_weight = append_weight
        
        self.alpha = alpha
        self.model_dir = model_dir

        if self.train_dataset and self.val_dataset:
            self._make_loader()
            self.num_train_steps = int(
                len(train_dataset) / self.batch_size * self.epochs
            )
        else:
            self.num_train_steps = 0
        self._get_optimizer(self.lr)

        if resume or checkpoint:
            self._load_model()

        tag_weights = torch.ones(
            self.tag_size, dtype=torch.float).to(self.device)
        tag_weights[-2] = self.change_weight
        tag_weights[-1] = self.append_weight

        self.tagger_loss = nn.CrossEntropyLoss(weight=tag_weights)
        self.generation_loss = nn.NLLLoss(ignore_index=0)

        # self.eval_tokenizer = GPT2Tokenizer.from_pretrained('danghuy1999/gpt2-viwiki')
        # self.eval_lm = GPT2LMHeadModel.from_pretrained('danghuy1999/gpt2-viwiki').to(device)

    def _load_model(self):
        assert os.path.exists(
            self.checkpoint), f"Checkpoint not found at {self.checkpoint}"
        # self.model.load_state_dict(torch.load(self.checkpoint))
        model_state = torch.load(self.checkpoint, map_location=self.device)
        self.model.load_state_dict(model_state["model"])
        sch_dict = model_state['scheduler']
        # total_epochs = self.epochs - self.resume_epoch
        # sch_dict['total_steps'] = sch_dict['total_steps'] + total_epochs * int(len(self.train_dataloader))
        self.scheduler.load_state_dict(sch_dict)
        self.optimizer.load_state_dict(model_state["optimizer"])

    def _make_loader(self):
        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            collate_fn=MyCollate(pad_id=0)
        )

        self.val_dataloader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            collate_fn=MyCollate(pad_id=0)
        )

    def _get_label(self, tag_output: Tensor, decoded_output: Tensor):
        pred_tag = torch.argmax(torch.softmax(tag_output, dim=-1), dim=-1)
        pred_decoded = torch.argmax(decoded_output, dim=-1).squeeze(1)
        return pred_tag, pred_decoded

    def _save_model_step(self, step):
        model_states = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict()
        }
        print(f'Saving at step {step}...')
        torch.save(model_states, self.model_dir +
                   f"model_step.{step}.pth")

    def _get_optimizer(self, lr):
        self.optimizer = AdamW(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, 
                                                            max_lr=0.01,
                                                            epochs=self.epochs,
                                                            steps_per_epoch=len(self.train_dataloader))
        # param_optimizer = list(self.model.named_parameters())
        # no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        # optimizer_parameters = [
        #     {
        #         "params": [
        #             p for n, p in param_optimizer if not any(
        #                 nd in n for nd in no_decay
        #             )
        #         ],
        #         "weight_decay": 0.001,
        #     }
        # ]

        # self.optimizer = AdamW(optimizer_parameters, lr=lr)
        # self.scheduler = get_linear_schedule_with_warmup(
        #     self.optimizer,
        #     num_warmup_steps=0,
        #     num_training_steps=self.num_train_steps
        # )
        self.scaler = torch.cuda.amp.GradScaler()

    def _save_model(self, epoch: int, history: dict):
        model_states = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict()
        }
        print(f'Saving epoch {epoch}...')
        logger.info(f'Saving epoch {epoch}...')
        torch.save(model_states, self.model_dir +
                   f"model_new_align/model{epoch}.pth")
        # print(f"Save model {epoch} done")
        with open(self.model_dir + f"model_new_align/history{epoch}.pkl", "wb") as file:
            pickle.dump(history, file)

    def _train_step(self):
        train_loss = 0
        self.model.train()

        with tqdm(enumerate(self.train_dataloader), unit="batch") as tepoch:
            for idx, data in tepoch:
                try:
                    self.optimizer.zero_grad()
                    with torch.cuda.amp.autocast():
                        word_errors = data["word_errors"].to(self.device)
                        char_errors = data["char_errors"].to(self.device)
                        targets = data["targets"].to(self.device)
                        source_splits = data["source_splits"]

                        tag_output, decoded_output = self.model(src_word_error_ids=word_errors,
                                                                src_char_ids=char_errors,
                                                                target=targets,
                                                                source_splits=source_splits)

                        result = self._calculate_loss_and_score(
                            tag_output, decoded_output, targets)

                    loss = result["total_loss"]
                    tag_loss = result["tag_loss"]
                    gen_loss = result["gen_loss"]


                    train_loss += loss.item()

                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    # loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_clipnorm)
                    self.scaler.step(self.optimizer)
                    # self.optimizer.step()
                    scale = self.scaler.get_scale()
                    self.scaler.update()

                    if scale > self.scaler.get_scale():
                        self.scheduler.step()

                    
                    tepoch.set_postfix(loss=loss.item(),
                                       tag_loss=tag_loss.item(),
                                       gen_loss=gen_loss.item())

                    if idx != 0 and idx % self.logging_steps == 0:
                        self._save_model_step(idx)
                        logger.info(f"{'-'*20} At step {idx} {'-'*20}")
                        logger.info(f"Loss: {loss.item()}")
                        logger.info(f"Gen Loss: {gen_loss.item()}")
                        logger.info(f"Tag Loss: {tag_loss.item()}")

                except Exception as e:
                    print(f"Issues with batch {idx} with exception {e}")
                    logger.warning("BUG")
                    logger.warning(e)
                    logger.info("-"*50)
                    logger.info(data["correct_sentences"])
                    logger.info(data["incorrect_sentences"])

        return train_loss / len(self.train_dataloader)

    def train(self):
        unimproved_counts = 0

        best_eval_loss = 100000
        self.model.zero_grad()
        history = defaultdict(list)
        for epoch in range(self.resume_epoch, self.epochs):
            logger.info("Start training")
            # start_time = time.time()
            self.epoch_temp = epoch

            train_loss = self._train_step()
            eval_loss, eval_tag_losses, eval_gen_losses, f1, f_beta, recall, precision = self._eval()

            history['train_loss'].append(train_loss)
            history['eval_loss'].append(eval_loss)
            history['eval_tag_losses'].append(eval_tag_losses)
            history['eval_gen_losses'].append(eval_gen_losses)

            print("*"*50)
            print(
                f"Epoch: {epoch} --- Train loss: {train_loss}\tVal loss: {eval_loss}\tVal tag loss: {eval_tag_losses}\tVal gen loss: {eval_gen_losses}")
            print(
                f"Val precision: {precision}\tVal recall: {recall}\tVal f1: {f1}\tVal f0.5: {f_beta}")
            logger.info(
                f"Epoch: {epoch} --- Train loss: {train_loss}\tVal loss: {eval_loss}\tVal tag loss: {eval_tag_losses}\tVal gen loss: {eval_gen_losses}")
            logger.info(
                f"Val precision: {precision}\tVal recall: {recall}\tVal f1: {f1}\tVal f0.5: {f_beta}")

            if eval_loss < best_eval_loss:
                history["best_loss"] = best_eval_loss
                best_eval_loss = eval_loss
                self._save_model(epoch, history)

    def _calculate_loss_and_score(self, 
                                tag_output, 
                                decoded_output, 
                                targets,
                                eval=False):
        result = {}

        tag_labels = targets[..., 0]

        if eval:
            pred_tags, pred_decoded = self._get_label(tag_output, decoded_output)
            pred_tags = torch.flatten(pred_tags).detach().tolist()
            true_tags = torch.flatten(tag_labels).detach().tolist()

            f1 = f1_score(true_tags, pred_tags, average="macro")
            f_beta = fbeta_score(true_tags, pred_tags, average='macro', beta=0.5)
            recall = recall_score(true_tags, pred_tags, average="macro")
            precision = precision_score(true_tags, pred_tags, average="macro")

            result["tag_f1"] = f1
            result["tag_f_beta"] = f_beta
            result["tag_recall"] = recall
            result["tag_precision"] = precision

        tag_decoded = []
        for batch_idx, tag in enumerate(tag_labels):
            for idx, label in enumerate(tag):
                if label in [2, 3]:
                    tag_decoded.append(
                        targets[batch_idx, idx, 2:].unsqueeze(0))

        tag_decoded = torch.cat(tag_decoded, dim=0)
  
        tag_loss = self.tagger_loss(tag_output.view(-1, self.tag_size),
                                    tag_labels.view(-1))

        gen_loss = F.nll_loss(decoded_output.view(-1, self.n_words),
                              tag_decoded.view(-1),
                              ignore_index=0)

        total_loss = self.alpha * tag_loss + gen_loss

        result.update({
            "gen_loss": gen_loss,
            "tag_loss": tag_loss,
            "total_loss": total_loss,
        })
        return result

    def infer(self, source, max_add_len=2, n_beams=1):
        self.model.eval()
        assert self.tokenizer and self.char_tokenizer, "Tokenizer not found"

        infer_dataset = InferDataset(source,
                                     self.tokenizer,
                                     self.char_tokenizer,
                                     self.bos_token_id,
                                     self.eos_token_id)

        infer_dataloader = DataLoader(infer_dataset,
                                      self.batch_size,
                                      num_workers=0,
                                      collate_fn=InferCollate(pad_id=0))
        results = []
        for _, data in enumerate(infer_dataloader):
            word_errors = data["word_errors"].to(self.device)
            char_errors = data["char_errors"].to(self.device)
            source_splits = data["source_splits"]
            source_sentences = data["source_sentences"]
            untok_sentences = data["untok_sentences"]
            ner_tags = data["ner_tags"]

            with torch.no_grad():
                result = self.model.infer(max_add_len=max_add_len,
                                          src_word_error_ids=word_errors,
                                          src_char_ids=char_errors,
                                          source_splits=source_splits,
                                          n_beams=n_beams)
                results.append(result)

        return self._process_infer_result(results, source_sentences, untok_sentences, ner_tags)

    def _process_infer_result(self, results, source_sentences, untok_sentences, ner_tags):
        predictions = []
        edits = []
        # print(ner_tags)

        for res in results:
            tag_pred = res["tag_pred"].detach().numpy()
            if "gen_results" in res:
                gen_results = res["gen_results"]
                preds = gen_results["preds"].detach().tolist()
                bidx_sidx_to_idx = gen_results["bidx_sidx_to_idx"]

                tokens = [self._process_infer_tokens(pred) for pred in preds]
                edit = self.tokenizer.sequences_to_texts(tokens)
                edits.extend(edit)

                delete_idx = np.where(tag_pred == 0)
                for sent_idx, token_idx in zip(delete_idx[0], delete_idx[1]):
                    if ner_tags[sent_idx][token_idx] == "O":
                        try:
                            int(source_sentences[sent_idx][token_idx])
                            tag_pred[sent_idx][token_idx] = TAGS["KEEP"]
                            continue
                        except:
                            # org_sentence = " ".join(source_sentences[sent_idx])
                            # tmp_sentence = source_sentences[sent_idx].copy()
                            # tmp_sentence[token_idx] = ""
                            # corrected_sentence = " ".join(tmp_sentence)
                            
                            # org_score = self._score_sentence([org_sentence]).tolist()[0]
                            # corrected_score = self._score_sentence([corrected_sentence]).tolist()[0]

                            # if corrected_score > org_score:
                            source_sentences[sent_idx][token_idx] = ""
                            # else:
                            #     tag_pred[sent_idx][token_idx] = TAGS["KEEP"]

                for (sent_idx, token_idx), pred_idx in bidx_sidx_to_idx.items():
                    if tag_pred[sent_idx][token_idx] == TAGS["CHANGE"]:
                        if ner_tags[sent_idx][token_idx] == "O":
                            try:
                                int(source_sentences[sent_idx][token_idx])
                                tag_pred[sent_idx][token_idx] = TAGS["KEEP"]
                                continue
                            except:
                                # org_sentence = " ".join(source_sentences[sent_idx])
                                # edit_text = " ".join(
                                #     self.tokenizer.sequences_to_texts([tokens[pred_idx]]))
                                # tmp_sentence = source_sentences[sent_idx].copy()
                                # tmp_sentence[token_idx] = edit_text
                                # corrected_sentence = " ".join(tmp_sentence)
                                
                                # org_score = self._score_sentence([org_sentence]).tolist()[0]
                                # corrected_score = self._score_sentence([corrected_sentence]).tolist()[0]

                                # if corrected_score > org_score:
                                    # source_sentences[sent_idx][token_idx] = edit_text
                                # else:
                                    # tag_pred[sent_idx][token_idx] = TAGS["KEEP"]
                                source_sentences[sent_idx][token_idx] = " ".join(
                                    self.tokenizer.sequences_to_texts([tokens[pred_idx]]))

                    elif tag_pred[sent_idx][token_idx] == TAGS["APPEND"]:
                        if ner_tags[sent_idx][token_idx] == "O":
                            try:
                                int(source_sentences[sent_idx][token_idx])
                                tag_pred[sent_idx][token_idx] = TAGS["KEEP"]
                                continue
                            except:
                                # org_sentence = " ".join(source_sentences[sent_idx])
                                # edit_text = " " + " ".join(self.tokenizer.sequences_to_texts([tokens[pred_idx]]))

                                # tmp_sentence = source_sentences[sent_idx].copy()
                                # tmp_sentence[token_idx] += edit_text
                                # corrected_sentence = " ".join(tmp_sentence)
                                
                                # org_score = self._score_sentence([org_sentence]).tolist()[0]
                                # corrected_score = self._score_sentence([corrected_sentence]).tolist()[0]

                                # if corrected_score > org_score:
                            # source_sentences[sent_idx][token_idx] += edit_text
                                # else:
                                #     tag_pred[sent_idx][token_idx] = TAGS["KEEP"]
                                source_sentences[sent_idx][token_idx] += " " + " ".join(
                                    self.tokenizer.sequences_to_texts([tokens[pred_idx]]))

            final_res = [" ".join(sentence) for sentence in source_sentences]
            final_res = [re.sub(r"\s+", " ", res) for res in final_res]
            predictions.extend(final_res)
        return tag_pred, predictions, edits

    def _process_infer_tokens(self, tokens):
        final_result = []
        for token in tokens:
            if token not in [self.bos_token_id, self.eos_token_id, self.pad_id]:
                final_result.append(token)

        return final_result

    def _eval(self):
        self.model.eval()

        eval_loss = []
        eval_tag_losses = []
        eval_gen_losses = []
        eval_f1 = []
        eval_f_beta = []
        eval_recall = []
        # eval_acc = []
        eval_precision = []
        logger.info(f"{'-'*20} RUNNING EVALUATION {'-'*20}")
        # for idx, data in tqdm(enumerate(self.val_dataloader)):
        with tqdm(enumerate(self.val_dataloader), unit="batch") as tepoch:
            for idx, data in tepoch:
                try:
                    word_errors = data["word_errors"].to(self.device)
                    char_errors = data["char_errors"].to(self.device)
                    targets = data["targets"].to(self.device)
                    source_splits = data["source_splits"]
                    # correct_sentences = data["correct_sentences"]

                    with torch.no_grad():
                        tag_output, decoded_output = self.model(src_word_error_ids=word_errors,
                                                                src_char_ids=char_errors,
                                                                target=targets,
                                                                source_splits=source_splits)

                        result = self._calculate_loss_and_score(
                            tag_output, decoded_output, targets, eval=True)

                        loss = result["total_loss"]
                        tag_loss = result["tag_loss"]
                        gen_loss = result["gen_loss"]
                        f1 = result["tag_f1"]
                        f_beta = result["tag_f_beta"]
                        recall = result["tag_recall"]
                        precision = result["tag_precision"]

                        eval_loss += [loss.mean().item()]
                        eval_tag_losses += [tag_loss.mean().item()]
                        eval_gen_losses += [gen_loss.mean().item()]
                        eval_f1.append(f1)
                        eval_f_beta.append(f_beta)
                        eval_recall.append(recall)
                        eval_precision.append(precision)

                        tepoch.set_postfix(loss=loss.item(),
                                           tag_loss=tag_loss.item(),
                                           gen_loss=gen_loss.item(),
                                           f1=f1,
                                           f_beta=f_beta,
                                           recall=recall,
                                           precision=precision)
                        # acc=acc)
                except Exception as e:
                    print(f"Issues with batch {idx} with exception {e}")
                    logger.warning("BUG")
                    logger.warning(e)
                    logger.info("-"*50)
                    logger.info(data["correct_sentences"])
                    logger.info(data["incorrect_sentences"])
                    break
                    # continue
                    # print("-"*20 + "Continue" + "-"*20)

        eval_loss = np.mean(eval_loss)
        eval_tag_losses = np.mean(eval_tag_losses)
        eval_gen_losses = np.mean(eval_gen_losses)
        eval_f1 = np.mean(eval_f1)
        eval_f_beta = np.mean(eval_f_beta)
        eval_recall = np.mean(eval_recall)
        eval_precision = np.mean(eval_precision)

        return eval_loss, eval_tag_losses, eval_gen_losses, eval_f1, eval_f_beta, eval_recall, eval_precision

    # def get_gpt2_loss(self, input_ids, attention_mask, labels):
    #     with torch.no_grad():
    #         outputs = self.eval_lm(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    #         lm_logits = outputs[1] #[bsize, seqlen, vocab]
    #         if labels is not None:
    #             shift_logits = lm_logits[..., :-1, :].contiguous()
    #             shift_labels = labels[..., 1:].contiguous()
    #             shift_mask = attention_mask[..., 1:].contiguous()
    #             loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    #             bsize, seqlen = input_ids.size()
    #             loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).view(bsize, seqlen-1)
    #             loss = (loss * shift_mask).sum(dim=1) #[bsize, ]
    #         return loss

    # def _score_sentence(self, sents):
    #     MAX_LENGTH = 100
    #     assert isinstance(sents, list)
    #     _sents = [self.eval_tokenizer.bos_token + s for s in sents]
    #     inputs = self.eval_tokenizer(_sents, return_tensors="pt")
    #     if inputs['input_ids'].size(1) > MAX_LENGTH:
    #         return None
    #     # if device:
    #     inputs = {k: v.to(self.device) for k, v in inputs.items()}
    #     loss = self.get_gpt2_loss(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=inputs['input_ids'])
    #     logps = - loss.detach().cpu()
    #     return logps
