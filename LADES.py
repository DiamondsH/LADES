import copy
import pickle
import re
import sys, os, dgl, random
import time

from transformers import TextGenerationPipeline, GPT2Tokenizer, GPT2LMHeadModel, pipeline

from Data.AmazonLoader import transIrregularWord

from DomainAdaptationTrainer.Utils.meta_learning_framework import MetaLearningFramework

from backpack import extend, extensions

from cockpit import Cockpit, CockpitPlotter, quantities
from cockpit.utils import schedules
# from test_gsnr.test_mean_gsnr import AutogradMeanGSNR
# Auto = AutogradMeanGSNR()



sys.path.append("..")

quantities = [
    # quantities.GradNorm(schedules.linear(interval=1)),
    # quantities.Distance(schedules.linear(interval=1)),
    # quantities.UpdateSize(schedules.linear(interval=1)),
    # quantities.HessMaxEV(schedules.linear(interval=3)),
    quantities.MeanGSNR(schedules.linear(interval=1)),
    # quantities.GradHist1d(schedules.linear(interval=10), bins=10),
]


from prefetch_generator import background
from BaseModel.BiGCN_Utils.RumorDetectionBasic import BaseEvaluator
from BaseModel.BiGCN_Utils.GraphRumorDect import BiGCNRumorDetecV2
from Data.BiGCN_Dataloader import BiGCNTwitterSet, FastBiGCNDataset, MetaMCMCDataset, load_data_all, load_data_twitter15
from BaseModel.modeling_bert import *
from transformers.models.bert import BertConfig, BertTokenizer
import torch, torch.nn as nn
from typing import List
from torch.utils.data import Dataset
from Data.BiGCN_Dataloader import load_data
from BaseModel.BiGCN_Utils.RumorDetectionBasic import RumorBaseTrainer
import pdb

torch.set_printoptions(threshold=torch.inf)

# assert torch.cuda.is_available() and torch.cuda.device_count() == 4


class SentBert(nn.Module):
    def __init__(self, bertPath):
        super(SentBert, self).__init__()
        self.bert_config = BertConfig.from_pretrained(bertPath, num_labels=2)
        self.tokenizer = BertTokenizer.from_pretrained(bertPath)
        # self.model = BertModel.from_pretrained(bertPath, config=self.bert_config).to(torch.device('cuda:0'))
        self.model = nn.DataParallel(
            BertModel.from_pretrained(bertPath, config=self.bert_config).to(torch.device('cuda:0')),
            # device_ids=[0, 1, 2, 3],
            device_ids = [0]
        )

    def text_to_batch_transformer(self, text: List):
        """Turn a piece of text into a batch for transformer model

        :param text: The text to tokenize and encode
        :param tokenizer: The tokenizer to use
        :param: text_pair: An optional second string (for multiple sentence sequences)
        :return: A list of IDs and a mask
        """
        max_len = self.tokenizer.max_len if hasattr(self.tokenizer, 'max_len') else self.tokenizer.model_max_length
        items = [self.tokenizer.encode_plus(sent, text_pair=None, add_special_tokens=True, max_length=max_len,
                                            return_length=False, return_attention_mask=True,
                                            return_token_type_ids=True)
                 for sent in text]
        input_ids = [item['input_ids'] for item in items]
        masks = [item['attention_mask'] for item in items]
        seg_ids = [item['token_type_ids'] for item in items]
        max_length = max([len(i) for i in input_ids])

        input_ids = torch.tensor([(i + [0] * (max_length - len(i))) for i in input_ids], device=torch.device('cuda:0'))
        masks = torch.tensor([(m + [0] * (max_length - len(m))) for m in masks], device=torch.device('cuda:0'))
        seg_ids = torch.tensor([(s + [0] * (max_length - len(s))) for s in seg_ids], device=torch.device('cuda:0'))
        return input_ids, masks, seg_ids

    def forward(self, sents):
        input_ids, masks, seg_ids = self.text_to_batch_transformer(sents)
        encoder_dict = self.model.forward(
            input_ids=input_ids,
            attention_mask=masks,
            token_type_ids=seg_ids
        )
        return encoder_dict.pooler_output  # .unsqueeze(0)

    def save_model(self, model_file):
        torch.save(self.model.module.state_dict(), model_file)

    def load_model(self, pretrained_file):
        self.model.module.load_state_dict(torch.load(pretrained_file))


class GraphEncoder(nn.Module):
    def __init__(self, config, num_hidden_layers):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(num_hidden_layers)])

    def set_aug_type(self, aug_type):
        for i in range(self.config.num_hidden_layers):
            self.layer[i].aug_type = aug_type

    def forward(
            self,
            hidden_states,
            root_idxs,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        assert hidden_states.dim() == 3
        lens = root_idxs + [hidden_states.data.size(1)]
        eye_mtx = torch.eye(len(root_idxs), device=hidden_states.data.device)
        trans_mtx = torch.stack([eye_mtx[jj - 1] for jj in range(1, len(root_idxs) + 1, 1) \
                                 for _ in range(lens[jj - 1], lens[jj])])

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if getattr(self.config, "gradient_checkpointing", False) and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                        "`use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            new_hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

            # hidden_states = #torch.matmul(trans_mtx, hidden_states[root_idxs]) \
            #                 + new_hidden_states
            hidden_states = torch.matmul(trans_mtx, hidden_states.squeeze(0)[root_idxs]).unsqueeze(0) \
                            + new_hidden_states

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class TwitterTransformer(BiGCNRumorDetecV2):
    def __init__(self, sent2vec, prop_model, rdm_cls, **kwargs):
        super(TwitterTransformer, self).__init__(sent2vec, prop_model, rdm_cls)
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise Exception(f"Attribute '{k}' is not a valid attribute of DaMSTF")
            self.__setattr__(k, v)
   

    def Batch2Vecs(self, batch):
        sentences = [text for s_list in batch[0] for text in s_list]
        assert len(sentences) == sum(batch[1])
        A_TD: torch.Tensor = batch[2].bool().float()
        A_BU: torch.Tensor = batch[3].bool().float()
        adj_mtx = (A_TD + A_BU).__ne__(0.0).float()
        attn_mask = (-10000 * (1.0 - adj_mtx)).unsqueeze(0).unsqueeze(0)
        inputs = self.sent2vec(sentences)
        num_nodes: List = batch[1]
        root_idxs = [sum(num_nodes[:idx]) for idx in range(len(num_nodes))]
        rst = self.prop_model(inputs.unsqueeze(0), attention_mask=attn_mask, root_idxs=root_idxs)
        hiddens = rst[0].squeeze(0)
        return hiddens[root_idxs]

    def meta_lossAndAcc(self, valid_data, temperature, label_weight):
        loss, acc = self.lossAndAcc(valid_data)
        return loss, acc

    def step(self, step_size): 
        for n, p in self.named_parameters():
            p.data = p.data - step_size * p.grad

    def lossAndAcc(self, batch, temperature=1.0, label_weight: torch.Tensor = None, reduction='mean'):

        preds = self.predict(batch, temperature=temperature)
        # 检查是否有可用的GPU
        epsilon = torch.ones_like(preds) * 1e-8
        preds = (preds - epsilon).abs()  # to avoid the prediction [1.0, 0.0], which leads to the 'nan' value in log operation
        labels = batch[-4].to(preds.device)
        loss, acc = self.loss_func(preds, labels, label_weight=label_weight, reduction=reduction)
        return loss, acc
    

    def get_CrossEntropyLoss(self,batch):
        seq_outs = self.Batch2Vecs(batch)
        logits = self.rdm_cls(seq_outs)
        labels = batch[-4].to(logits.device)
        loss_fn = extend(torch.nn.CrossEntropyLoss(reduction="mean"))
        loss=loss_fn(logits,labels)
        return loss
    
    def get_MetaLearningCrossEntropyLoss(self,batch,weights):
        seq_outs = self.Batch2Vecs(batch)
        logits = self.rdm_cls(seq_outs)
        labels = batch[-4].to(logits.device)
        loss_fn = extend(torch.nn.CrossEntropyLoss(reduction="mean"))
        loss=loss_fn(logits,labels)
        return loss
    
    def loss_func(self, preds: torch.Tensor, labels: torch.Tensor, label_weight=None, reduction='none'):
        # print("preds.shape",preds.shape)
        # print("labels.shape", labels.shape)
        if labels.dim() == 3:
            loss, acc = self.expandCrossEntropy(preds, labels, label_weight, reduction)
        elif labels.dim() == 2:
            # labels = labels.unsqueeze(dim=1)
            loss, acc = self.expandCrossEntropy(preds, labels, label_weight, reduction)
        elif labels.dim() == 1:
            loss = F.nll_loss(preds.log(), labels, weight=label_weight, reduction=reduction)
            acc_t = ((preds.argmax(dim=1) - labels).__eq__(0).float().sum()) / len(labels)
            acc = acc_t.data.item()
        else:
            raise Exception("weird label tensor!")
        return loss, acc


def obtain_Transformer(bertPath, device=None):
    if device is None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
    lvec = SentBert(bertPath)
    prop = GraphEncoder(lvec.bert_config, 2).to(device)
    cls = extend(nn.Linear(768, 2)).to(device)
    BiGCN_model = TwitterTransformer(lvec, prop, cls)
    return BiGCN_model


class TransformerEvaluator(BaseEvaluator):
    def __init__(self, dataset: BiGCNTwitterSet, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.labelTensor = dataset.labelTensor()

    def collate_fn(self, items):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
        sent_list = [item.text for item in items]
        TD_graphs = [item.g_TD for item in items]
        BU_graphs = [item.g_BU for item in items]
        labels = [item.data_y for item in items]
        topic_labels = [item.topic_label for item in items]
        num_nodes = [g.num_nodes() for g in TD_graphs]
        big_g_TD = dgl.batch(TD_graphs)
        big_g_BU = dgl.batch(BU_graphs)
        A_TD = big_g_TD.adjacency_matrix().to_dense().to(device)
        A_BU = big_g_BU.adjacency_matrix().to_dense().to(device)
        return sent_list, num_nodes, A_TD, A_BU, \
            torch.tensor(labels), torch.tensor(topic_labels)

    def dataset2dataloader(self):
        idxs = [*range(len(self.dataset))]

        @background(max_prefetch=5)
        def dataloader():
            for start in range(0, len(self.dataset), self.batch_size):
                batch_idxs = idxs[start:min(start + self.batch_size, len(self.dataset))]
                items = [self.dataset[index] for index in batch_idxs]
                yield self.collate_fn(items)

        return dataloader()


class BiGCNTrainer(RumorBaseTrainer):
    def __init__(self, log_dir, tokenizer, **kwargs):
        super(BiGCNTrainer, self).__init__()
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise Exception(f"Attribute '{k}' is not a valid attribute of DaMSTF")
            self.__setattr__(k, v)

        self.running_dir = log_dir
        if not os.path.exists(self.running_dir):
            os.system(f"mkdir {self.running_dir}")
        self.tokenizer = tokenizer
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def collate_fn(self, items):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
        sents = [item.text for item in items]
        TD_graphs = [item.g_TD for item in items]
        BU_graphs = [item.g_BU for item in items]
        labels = [item.data_y for item in items]
        topic_labels = [item.topic_label for item in items]
        num_nodes = [g.num_nodes() for g in TD_graphs]
        big_g_TD = dgl.batch(TD_graphs)
        big_g_BU = dgl.batch(BU_graphs)
        A_TD = big_g_TD.adjacency_matrix().to_dense().to(device)
        A_BU = big_g_BU.adjacency_matrix().to_dense().to(device)
        return sents, num_nodes, A_TD, A_BU, \
            torch.tensor(labels), torch.tensor(topic_labels)

    def trainset2trainloader(self, dataset: Dataset, shuffle=False, batch_size=32):
        return self.dataset2dataloader(dataset, shuffle, batch_size)

    def dataset2dataloader(self, dataset: BiGCNTwitterSet, shuffle=False, batch_size=32):
        if shuffle:
            idxs = random.sample(range(len(dataset)), len(dataset)) * 2
        else:
            idxs = [*range(len(dataset))] * 2

        @background(max_prefetch=5)
        def dataloader():
            for start in range(0, len(dataset), batch_size):
                batch_idxs = idxs[start:start + batch_size]
                items = [dataset[index] for index in batch_idxs]
                yield self.collate_fn(items)

        return dataloader()

    def fit(self, model: TwitterTransformer, train_set, dev_eval=None, test_eval=None, batch_size=5,
            grad_accum_cnt=4,
            valid_every=100, max_epochs=10, learning_rate=5e-3, model_file=""):
        best_valid_acc, counter = 0.0, 0
        sum_loss, sum_acc = 0.0, 0.0
        optim = model.obtain_optim(learning_rate * 1.0 / grad_accum_cnt)
        optim.zero_grad()
        loss_list = []
        # model.load_model("./ferguson/BiGCN_ferguson_0.817.pkl")
        # model.load_model("./ottawashooting/BiGCN_ottawashooting_0.72.pkl")
        # model.load_model("./charliehebdo/BiGCN_charliehebdo_0.85.pkl")
        
        # model = extend(model)  
        # model=model.cuda()
        # parameters=model.parameters().cuda()

        cockpit = Cockpit(model.rdm_cls.parameters(), quantities=quantities)
        plotter = CockpitPlotter()
        max_steps, global_step = 500, 0

        for epoch in range(max_epochs):
            train_loader = self.trainset2trainloader(model, train_set, shuffle=True, batch_size=batch_size)
            if global_step >= max_steps:
                print("traing out end")
                break

            for batch in train_loader:
                counter += 1
                # model.lossAndAcc=extend(model.lossAndAcc)
                loss, acc = model.lossAndAcc(batch)
                losses=model.get_CrossEntropyLoss(batch)
                # print("loss",loss)
                # print("losses",losses)
                # losses, acces = extend(model.lossAndAcc(batch))

                # meangsnr = Auto._compute(global_step, model.parameters(), loss)
                # print("meangsnr", meangsnr)
 
                # backward pass
                with cockpit(
                    global_step,
                    extensions.DiagHessian(),  # Other BackPACK quantities can be computed as well
                    info={
                        "batch_size": batch_size,
                        # "individual_losses": losses,
                        "loss": losses,
                        "optimizer": optim,
                    },
                ):
                    losses.backward(create_graph=cockpit.create_graph(global_step))
                    # losses.backward()
                    
                # loss.backward()
                # loss.backward()
                torch.cuda.empty_cache()
                sum_loss += loss.data.item()
                sum_acc += acc
                if counter % grad_accum_cnt == 0:
                    optim.step()
                    global_step += 1
                    print(f"Step: {global_step:5d} | Loss: {loss.item():.4f}")
                    if global_step % 5 == 0:
                        plotter.plot(
                            cockpit,
                            savedir=get_logpath(suffix="ottawashooting"),
                            show_plot=False,
                            save_plot=True,
                            savename_append=str(global_step),
                        )
                    if global_step >= max_steps:
                        print("traing in end")
                        break

                    optim.zero_grad()
                    mean_loss, mean_acc = sum_loss / grad_accum_cnt, sum_acc / grad_accum_cnt
                    loss_list.append(mean_loss)
                    if len(loss_list) > 20:
                        loss_list.pop(0)
                    print(
                        '%6d  [%3d | %3d], loss/acc = %6.8f/%6.7f, loss_mean/std=%6.7f/%6.7f, best_valid_acc:%6.7f ' % (
                            counter, epoch, max_epochs,
                            mean_loss, mean_acc, np.mean(loss_list), np.std(loss_list),
                            best_valid_acc))
                    sum_loss, sum_acc = 0.0, 0.0

                if counter % (valid_every * grad_accum_cnt) == 0 and dev_eval is not None:
                    val_acc = dev_eval(model)
                    if val_acc > best_valid_acc:
                        best_valid_acc = val_acc
                        model.save_model(model_file)
        
        # Write Cockpit to json file.
        cockpit.write(get_logpath(suffix="ottawashooting"))

        # Plot results from file
        plotter.plot(
            get_logpath(suffix="ottawashooting0323"),
            savedir=get_logpath(suffix="ottawashooting0323"),
            show_plot=False,
            save_plot=True,
            savename_append="_final",
        )

        model.load_model(model_file)
        if test_eval is not None:
            test_acc = test_eval(model)
        else:
            test_acc = best_valid_acc
        if self.model_rename:
            self.RenameModel(model_file, test_acc)


class MetaBiGCNTrainer(BiGCNTrainer, MetaLearningFramework):
    def __init__(self, log_dir, tokenizer, **kwarg):
        super(MetaBiGCNTrainer, self).__init__(log_dir, tokenizer, **kwarg)
        self.meta_valid_bs = 16

    def cosine(self, vec1: torch.Tensor, vec2: torch.Tensor):
        return torch.dot(vec1, vec2) / (vec1.norm() * vec2.norm())

    def collate_fn(self, items):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
        sents = [item.text for item in items]
        TD_graphs = [item.g_TD for item in items]
        BU_graphs = [item.g_BU for item in items]
        labels = [item.data_y for item in items]
        topic_labels = [item.topic_label for item in items]
        num_nodes = [g.num_nodes() for g in TD_graphs]
        big_g_TD = dgl.batch(TD_graphs)
        big_g_BU = dgl.batch(BU_graphs)
        A_TD = big_g_TD.adjacency_matrix().to_dense().to(device)
        A_BU = big_g_BU.adjacency_matrix().to_dense().to(device)
        weights = torch.stack([item.weight for item in items])
        idxs = torch.tensor([item.index for item in items], dtype=torch.long,
                            device=device)
        return sents, num_nodes, A_TD, A_BU, torch.tensor(labels), torch.tensor(topic_labels), weights, idxs

    def darts_approximation(self, model: TwitterTransformer, valid_data, training_batch, temperature=1.0,
                            label_weight=None):
        model.zero_grad()
        meta_loss, meta_acc = self.gradOnMetaValSet(model, valid_data)
        if hasattr(self, 'meta_loss_list'):
            if len(self.meta_loss_list) >= 50:
                self.meta_loss_list.pop(0)
            self.meta_loss_list.append(meta_loss)
        else:
            self.meta_loss_list = [meta_loss]

        # self.meta_grad = model.grad_vec()
        print(f"======> Meta Validation ===> loss/acc = {meta_loss}/{meta_acc}")
        model.step(-1 * self.epsilon)  # theta^+
        with torch.no_grad():
            loss_1 = model.lossList(training_batch, temperature=1.0, label_weight=None)

        model.step(2 * self.epsilon)  # theta^-
        with torch.no_grad():
            loss_2 = model.lossList(training_batch, temperature=1.0, label_weight=None)

        model.step(-1 * self.epsilon)  # theta_hat
        grad = (loss_1 - loss_2) / (2 * self.epsilon)
        return grad, meta_loss

    def print_gradient_information(self, gradient: torch.Tensor):
        pass

    def meta_learning(self, model: TwitterTransformer, train_set, meta_dev,
                      dev_eval: TransformerEvaluator, test_eval: TransformerEvaluator, batch_size=32, grad_accum_cnt=4,
                      valid_every=100, max_epochs=10, learning_rate=5e-3, model_lr=2e-5, model_file="",
                      weight_lr=0.1, meta_step=10):

        best_valid_acc, counter = 0.0, 0
        sum_loss, sum_acc = 0.0, 0.0
        optim = model.obtain_optim(learning_rate * 1.0 / grad_accum_cnt)
        optim.zero_grad()
        loss_lists = []
        train_step = 0
        # model.load_model("./indomain/pheme/BiGCN_pheme_0.857.pkl")
        # model.load_model("./ferguson/BiGCN_ferguson_0.79.pkl")
        model.load_model("./ottawashooting/BiGCN_ottawashooting_0.878.pkl")
        cockpit = Cockpit(model.rdm_cls.parameters(), quantities=quantities)
        plotter = CockpitPlotter()
        max_steps, global_step = 50, 0
        for epoch in range(max_epochs):
            print("epoch", epoch)

            with torch.no_grad():
                logits = model.dataset_logits(meta_dev, temperature=1.0, batch_size=20)
                vals, indexs = logits.sort(dim=1)
                probs, preds = vals[:, -1], indexs[:, -1]
            train_loader = self.trainset2trainloader(model, train_set, shuffle=True, batch_size=batch_size)
            weights_list, idxs_list = [], []
            for batch in train_loader:
                counter += 1
                v_batch = meta_dev.collate_fn([
                    meta_dev[m_idx.data.item()] for m_idx in torch.multinomial(
                        torch.exp(-5.0 * probs), self.meta_valid_bs, replacement=False, generator=None
                    )
                ])
                init_state_dict = model.state_dict()
                idxs, weights = batch[-1], batch[-2]

                idxs_list.append(idxs)
                for mstep in range(meta_step):
                    if mstep != 0:
                        model.load_state_dict(init_state_dict)
                    loss_list = model.lossList(batch)
                    u = weights.sigmoid()
                    loss = (u * loss_list).sum()
                    loss.backward()
                    model.step(model_lr)  # theta_hat
                    model.zero_grad()
                    u_grads, meta_loss = self.darts_approximation(model, v_batch, batch, temperature=1.0,
                                                                  label_weight=None)
                    w_grads = u_grads * u * (1 - u)
                    normed_wGrads = -1 * (w_grads / (w_grads.norm(2) + 1e-10))
                    weights = weights - weight_lr * normed_wGrads
                weights_list.append(weights)
                loss, acc = model.lossAndAcc(batch)
                losses=model.get_CrossEntropyLoss(batch)

                normed_weights = weights / (weights.sum())
                loss = (normed_weights * loss).sum()        

                normed_weights = weights / (weights.sum())
                losses = (normed_weights * losses).sum()
                
                print("loss",loss)
                print("losses",losses)
                # sig_weights = torch.clamp(weights.sigmoid(), 0.5)
                 # backward pass
                with cockpit(
                    global_step,
                    extensions.DiagHessian(),  # Other BackPACK quantities can be computed as well
                    info={
                        "batch_size": batch_size,
                        # "individual_losses": losses,
                        "loss": losses,
                        "optimizer": optim,
                    },
                ):
                    losses.backward(create_graph=cockpit.create_graph(global_step))

                # loss.backward()
                torch.cuda.empty_cache()
                sum_loss += loss.data.item()
                sum_acc += acc
                train_step += 1
                if counter % grad_accum_cnt == 0:
                    optim.step()
                    global_step += 1
                    print(f"Step: {global_step:5d} | Loss: {loss.item():.4f}")
                    if global_step % 10 == 0:
                        plotter.plot(
                            cockpit,
                            savedir=get_logpath(),
                            show_plot=False,
                            save_plot=True,
                            savename_append=str(global_step),
                        )
                    optim.zero_grad()
                    mean_loss, mean_acc = sum_loss / grad_accum_cnt, sum_acc / grad_accum_cnt
                    loss_lists.append(mean_loss)
                    if len(loss_lists) > 20:
                        loss_lists.pop(0)
                    print(
                        '%6d  [%3d | %3d], loss/acc = %6.8f/%6.7f, loss_mean/std=%6.7f/%6.7f, best_valid_acc:%6.7f ' % (
                            counter, epoch, max_epochs,
                            mean_loss, mean_acc, np.mean(loss_lists), np.std(loss_lists),
                            best_valid_acc))
                    sum_loss, sum_acc = 0.0, 0.0

                if counter % (valid_every * grad_accum_cnt) == 0 and dev_eval is not None:
                    val_acc = dev_eval(model)
                    if val_acc > best_valid_acc:
                        best_valid_acc = val_acc
                        model.save_model(model_file)

                old_weights = train_set.instance_weights.clone()
                train_set.instance_weights[torch.cat(idxs_list)] = torch.squeeze(torch.cat(weights_list))
                self.print_divergence(train_set.instance_weights,
                                      old_weights)
        
        # Write Cockpit to json file.
        cockpit.write(get_logpath())

        # Plot results from file
        plotter.plot(
            get_logpath(),
            savedir=get_logpath(),
            show_plot=False,
            save_plot=True,
            savename_append="_final",
        )

        model.load_model(model_file)
        if test_eval is not None:
            test_acc = test_eval(model)
        else:
            test_acc = best_valid_acc
        print("test_acc",test_acc)
        if self.model_rename:
            self.RenameModel(model_file, test_acc)


    def validate_cpt(self,model: TwitterTransformer,test_eval: TransformerEvaluator):
        # model.load_model("./ottawashooting/BiGCN_ottawashooting_0.878.pkl")
        # model.load_model("./ferguson/BiGCN_ferguson_0.861.pkl")
        # model.load_model("./sydneysiege/BiGCN_sydneysiege_0.759.pkl")
        model.load_model("./charliehebdo/BiGCN_charliehebdo_0.84.pkl")
        test_acc = test_eval(model)
        print("test_acc",test_acc)


class MCMCBiGCNTrainer(MetaBiGCNTrainer, FastBiGCNDataset):
    def __init__(self, log_dir, tokenizer, beam_K, batch_dir, **kwargs):
        super(MCMCBiGCNTrainer, self).__init__(log_dir, tokenizer, **kwargs)
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise Exception(f"Attribute '{k}' is not a valid attribute of DaMSTF")
            self.__setattr__(k, v)

        self.running_dir = log_dir
        if not os.path.exists(self.running_dir):
            os.system(f"mkdir {self.running_dir}")
        self.tokenizer = tokenizer
        self.batch_dir = batch_dir
        self.beam_K = beam_K
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def trainset2trainloader(self, model: TwitterTransformer, dataset: MetaMCMCDataset, shuffle=False, batch_size=32):
        return self.dataset2dataloader(dataset, shuffle, batch_size)
        # return self.dataset2dataloader3()

    def dataset2dataloader(self, dataset: BiGCNTwitterSet, shuffle=False, batch_size=32):
        if shuffle:
            idxs = random.sample(range(len(dataset)), len(dataset)) * 2
        else:
            idxs = [*range(len(dataset))] * 2

        @background(max_prefetch=5)
        def dataloader():
            for start in range(0, len(dataset), batch_size):
                batch_idxs = idxs[start:start + batch_size]
                items = [dataset[index] for index in batch_idxs]
                yield self.collate_fn(items)

        return dataloader()

    def dataset2dataloader3(self):

        @background(max_prefetch=5)
        def AugBatchGenerator(aug_dir):
            batch_files = [fname for fname in os.listdir(aug_dir) if fname[-6:] == '.batch']
            print("len batch_files", len(batch_files))
            batch_files.sort()
            batch_files_len = len(batch_files)
            for _ in range(0, batch_files_len, 1):
                batch_cur = batch_files.pop(0)
                batch_path = os.path.join(aug_dir, batch_cur)
                with open(batch_path, 'rb') as fr:
                    data = pickle.load(fr)
                yield data

        return AugBatchGenerator(self.batch_dir)

    def dataset2dataloader2(self, model: TwitterTransformer, dataset: MetaMCMCDataset, shuffle=False, batch_size=32):
        if shuffle:
            idxs = random.sample(range(len(dataset)), len(dataset)) * 2
        else:
            idxs = [*range(len(dataset))] * 2

        tok = GPT2Tokenizer.from_pretrained("/data/run01/scz3924/hezj/MetaGenerator/MetaGraphAug/gpt2")
        # tok.model_max_length = 128
        gpt2 = GPT2LMHeadModel.from_pretrained("/data/run01/scz3924/hezj/MetaGenerator/MetaGraphAug/gpt2").cuda()
        from transformers import pipeline

        lm_generator = pipeline('text-generation', model=gpt2, tokenizer=tok,
                                device=0 if torch.cuda.is_available() else -1)

        tr_data_len = len(dataset)
        # model.load_model("./ferguson/BiGCN_ferguson_0.77.pkl")
        # model.load_model("./charliehebdo/BiGCN_charliehebdo_0.84.pkl")
        # model.load_model("./ottawashooting/BiGCN_ottawashooting_0.79.pkl")
        # model.load_model("./ottawashooting/BiGCN_ottawashooting_0.831.pkl")
        model.load_model("./sydneysiege/BiGCN_sydneysiege_0.72.pkl")



        # for start in range(0, len(dataset), batch_size):
        #     batch_idxs = idxs[start:start + batch_size]
        #     # new_path = self.get_new_model_path(self.running_dir)
        #     # if new_path is not None:
        #     #     model.load_model(new_path)
        #     self.MCMC_Augmentation(dataset, model, lm_generator, max_step=5, batch_idxs=batch_idxs,
        #                            batch_size=32, temperature=0.1,
        #                            tr_data_len=tr_data_len)
        # batch_idxs1 = random.sample(range(4581), 30)
        # batch_idxs2 = random.sample(range(4581,5800), 2)
        batch_idxs = random.sample(range(tr_data_len), batch_size)
        # batch_idxs = batch_idxs1+batch_idxs2
        # new_path = self.get_new_model_path(self.running_dir)
        # if new_path is not None:
        #     model.load_model(new_path)
        self.MCMC_Augmentation(dataset, model, lm_generator, max_step=10,early_stop_confidence=0.5, batch_idxs=batch_idxs,
                               batch_size=32, temperature=0.1,
                               tr_data_len=tr_data_len)


        # @background(max_prefetch=5)
        # def dataloader():
        #     for start in range(0, len(dataset), batch_size):
        #         batch_idxs = idxs[start:start + batch_size]
        #         new_path = self.get_new_model_path(self.running_dir)
        #         if new_path is not None:
        #             model.load_model(new_path)
        #         batch = self.MCMC_Augmentation(dataset, model, lm_generator, max_step=5, batch_idxs=batch_idxs,
        #                                        batch_size=32, temperature=0.1,
        #                                        tr_data_len=tr_data_len)
        #         if batch is None:
        #             continue
        #         else:
        #             yield batch
        #
        # return dataloader()

    def MCMC_Augmentation(self, trdataset: MetaMCMCDataset, model: TwitterTransformer,
                          lm_generator: pipeline,
                          max_step=5, early_stop_confidence=0.5, batch_idxs: List = None,
                          batch_size=10, temperature=1.0, kappa=0.5, tr_data_len=4600):
        dataset = copy.deepcopy(trdataset)
        if not hasattr(self, 'init_length'):
            self.init_length = len(dataset.data_ID)
        if batch_idxs is None:
            batch_idxs = random.sample(range(tr_data_len), batch_size)
        else:
            assert len(batch_idxs) == batch_size
        d_ID_list = [dataset.data_ID[idx] for idx in batch_idxs]
        try:
            # [copy.deepcopy(dataset.__getitem__(idx)) for idx in batch_idxs]
            new_data_list = [copy.deepcopy(dataset.data[d_ID]) for d_ID in d_ID_list]
        except:
            import pdb
            pdb.set_trace()
            raise

        timestamp = int(time.time())
        batch_IDs = [str(timestamp + kk) for kk in range(len(batch_idxs))]
        dataset.data_ID.extend(batch_IDs)
        for kk, bID in enumerate(batch_IDs):
            dataset.data[bID] = new_data_list[kk]
            dataset.g_TD[bID] = copy.deepcopy(dataset.g_TD[dataset.data_ID[batch_idxs[kk]]])
            dataset.g_BU[bID] = copy.deepcopy(dataset.g_BU[dataset.data_ID[batch_idxs[kk]]])
        dataset.data_len.extend(
            [dataset.data_len[idx] for idx in batch_idxs]
        )
        dataset.data_y.extend(
            [dataset.data_y[idx] for idx in batch_idxs]
        )
        batch_labels = torch.tensor(
            [dataset.data_y[idx] for idx in batch_idxs],
            device=self.device,
            dtype=torch.float32
        )
        texts_cur = [[" ".join(self.lemma(dataset.data[b_ID]['text'][j])) for j in range(dataset.data_len[index])]
                     for b_ID, index in zip(batch_IDs, batch_idxs)]
        TD_Graphs_cur = [copy.deepcopy(dataset.g_TD[dataset.data_ID[b_idx]]).add_self_loop() for b_idx in batch_idxs]
        BU_Graphs_cur = [copy.deepcopy(dataset.g_BU[dataset.data_ID[b_idx]]).add_self_loop() for b_idx in batch_idxs]
        confidence_cur,logits_cur = self.get_logits(model, texts_cur, TD_Graphs_cur, BU_Graphs_cur, batch_labels)
        print("start confidence calue： ",logits_cur[0].item())
        # print("start logits calue： ",logits_cur[0].item())
        MCMC_logits=[]
        MCMC_logits2 = []
        MCMC_logits.append(logits_cur[0].item())
        MCMC_logits2.append(logits_cur.sum(dim=0).item()/32)
        best_confidence = confidence_cur.clone()

        for step in range(max_step):
            if logits_cur.max().item() < early_stop_confidence:
                print("early Stop")
                print("MCMC_logits", MCMC_logits)
                print("MCMC_logits2 ", MCMC_logits2)
                return
            # if logits_cur[1].item()<(early_stop_confidence-0.3):
            #     print("early Stop")
            #     print("MCMC_logits", MCMC_logits)
            #     print("MCMC_logits2 ", MCMC_logits2)
            #     return
            items = [self.sample_func(td_tree, bu_tree, texts)
                     for texts, td_tree, bu_tree in zip(texts_cur, TD_Graphs_cur, BU_Graphs_cur)]
            prompt_sents = [item[1] for item in items if item[1] is not None]
            TD_Graphs_next = [item[3] for item in items]
            BU_Graphs_next = [item[4] for item in items]
            accept_term_2 = torch.tensor(
                [item[5] for item in items],
                dtype=torch.float32,
                device=self.device
            )

            reply_sents = self.sample_replies(prompt_sents, lm_generator)
            texts_next = [item[0] for item in items]
            c_idx = 0
            for kk, item in enumerate(items):
                if item[1] is not None:
                    texts_next[kk][item[2]] = reply_sents[c_idx]
                    c_idx += 1
            confidence_next,logits_next = self.get_logits(model, texts_next, TD_Graphs_next,
                                                BU_Graphs_next, batch_labels)
            # print("step: ",step,"current logits value：",confidence_next[0].item())
            print("step: ",step,"current logits value：",logits_next[0].item())
            MCMC_logits.append(logits_next[0].item())
            MCMC_logits2.append(logits_next.sum(dim=0).item()/32)

            delta_confidence = (confidence_next - confidence_cur) / temperature
            delta_confidence = torch.where(delta_confidence.__le__(3.0),
                                           delta_confidence,
                                           3.0 * torch.ones_like(delta_confidence, device=delta_confidence.device))
            accept_term_1 = torch.exp(delta_confidence)
            accept_term_3 = torch.pow(
                torch.exp(
                    torch.tensor(
                        [TD_Graphs_next[idx].num_nodes() * 1.0 / TD_Graphs_cur[idx].num_nodes()
                         for idx in range(batch_size)],
                        device=self.device,
                        dtype=torch.float32
                    )
                ),
                1.0 / kappa
            )
            accept_ratio = (accept_term_1 * accept_term_2) / accept_term_3
            print("accept_term_1 : ", accept_term_1)
            print("accept_term_2 : ", accept_term_2)
            print("accept_term_3 : ", accept_term_3)
            print("accept_ratio : ", accept_ratio.cpu().tolist())
            # accept_idxs = torch.arange(batch_size)[
            #     (accept_ratio.cpu() - torch.rand(batch_size)).__gt__(0.0)
            # ].tolist()

            accept_idxs = torch.arange(batch_size).tolist()

            for a_idx in accept_idxs:
                TD_Graphs_cur[a_idx] = TD_Graphs_next[a_idx]
                BU_Graphs_cur[a_idx] = BU_Graphs_next[a_idx]
                texts_cur[a_idx] = texts_next[a_idx]

            preserve_idx = torch.arange(batch_size)[
                (confidence_next - best_confidence).__gt__(0).cpu()
            ].tolist()
            for p_idx in preserve_idx:
                dataset.data[batch_IDs[p_idx]]['text'] = [
                    [item for item in re.split(r'[ ]+', s) if item.strip(" ") != '']
                    for s in texts_next[p_idx]]
                dataset.data_len[len(dataset.data_len) - batch_size + p_idx] = len(texts_next[p_idx])
                dataset.g_BU[batch_IDs[p_idx]] = BU_Graphs_next[p_idx]
                dataset.g_TD[batch_IDs[p_idx]] = TD_Graphs_next[p_idx]
            best_confidence = torch.where(confidence_next > best_confidence, confidence_next, best_confidence)
            confidence_cur = confidence_next
            logits_cur = logits_next
        # self.extract_augmented_batch(batch_size, dataset)
        print("MCMC_logits",MCMC_logits)
        print("MCMC_logits2 ",MCMC_logits2)

    def return_augmented_batch(self, batch_size, dataset):
        items = [dataset.__getitem__(index) for index in range(-1 * batch_size, 0, 1)]
        batch = self.collate_fn(items)
        for _ in range(batch_size):
            d_ID = dataset.data_ID.pop(-1)
            dataset.data_len.pop(-1)
            dataset.data_y.pop(-1)
            dataset.g_TD.pop(d_ID)
            dataset.g_BU.pop(d_ID)
            dataset.data.pop(d_ID)
        return batch

    def extract_augmented_batch(self, batch_size, dataset):
        items = [dataset.__getitem__(index) for index in range(-1 * batch_size, 0, 1)]
        batch = self.collate_fn(items)
        t = int(time.time())
        for _ in range(batch_size):
            d_ID = dataset.data_ID.pop(-1)
            dataset.data_len.pop(-1)
            dataset.data_y.pop(-1)
            dataset.g_TD.pop(d_ID)
            dataset.g_BU.pop(d_ID)
            dataset.data.pop(d_ID)
        with open(os.path.join(self.batch_dir, f"{t}.batch"), "wb") as fw:
            pickle.dump(batch, fw, protocol=pickle.HIGHEST_PROTOCOL)
        signal_file = os.path.join(self.batch_dir, f"{t}.ok")
        os.system(f"touch {signal_file}")

    def get_new_model_path(self, model_dir: str):
        print("model_dir", model_dir)
        model_path = os.path.join(model_dir, "BiGCN_charliehebdo.pkl")
        if os.path.exists(model_path):
            return model_path
        else:
            model_path = os.path.join(model_dir, "BiGCN_charliehebdo_0.86.pkl")
            return model_path

    def model_logits(self, model: TwitterTransformer, texts: List, TD_graphs: List, BU_graphs: List,
                     batch_labels: torch.Tensor, temperature=1.0):
        num_nodes = [g.num_nodes() for g in TD_graphs]
        big_g_TD = dgl.batch(TD_graphs)
        big_g_BU = dgl.batch(BU_graphs)
        A_TD = big_g_TD.adjacency_matrix().to_dense().to(self.device)
        A_BU = big_g_BU.adjacency_matrix().to_dense().to(self.device)
        with torch.no_grad():
            logits = model.predict(
                (texts, num_nodes, A_TD, A_BU),
                temperature=temperature
            )
        energy = (batch_labels * (logits.log().neg())).sum(dim=1)
        print("energy : ", energy)
        return energy,(logits*batch_labels).sum(dim=1)

    def get_logits(self, model: TwitterTransformer, texts: List, TD_graphs: List, BU_graphs: List,
                     batch_labels: torch.Tensor, temperature=1.0):
        num_nodes = [g.num_nodes() for g in TD_graphs]
        big_g_TD = dgl.batch(TD_graphs)
        big_g_BU = dgl.batch(BU_graphs)
        A_TD = big_g_TD.adjacency_matrix().to_dense().to(self.device)
        A_BU = big_g_BU.adjacency_matrix().to_dense().to(self.device)
        with torch.no_grad():
            logits = model.predict(
                (texts, num_nodes, A_TD, A_BU),
                temperature=temperature
            )
        energy = (batch_labels * (logits.log().neg())).sum(dim=1)
        logits_value=(batch_labels*logits).sum(dim=1)
        print("energy : ", energy)
        print("logits_value : ", logits_value)
        # return (batch_labels*logits).sum(dim=1)
        return energy,logits_value



    def sample_replies(self, sents, lm_generator: TextGenerationPipeline):
        replies_list = []
        lens = [len(sent.split()) for sent in sents]
        for idx in range(0, len(sents), 1):
            with torch.no_grad():
                rst = lm_generator([sents[idx]],
                                   num_return_sequences=self.beam_K,
                                   min_length=lens[idx] + 20,
                                   max_length=lens[idx] + 64,  # max length of the generated text
                                   return_full_text=False)
                replies = [item[0]['generated_text'] for item in rst]
                torch.cuda.empty_cache()
                replies_list.append(replies)
        sampled_replies = [transIrregularWord(random.sample(rl, 1)[0]) for rl in replies_list]
        return sampled_replies

    def sample_func(self, td_graph: dgl.DGLGraph, bu_graph: dgl.DGLGraph, texts: List):
        texts_new = copy.deepcopy(texts)
        if td_graph.num_nodes() == 1:
            aug_type, txt, prob, m_idx, new_td, new_bu = self.add_node(td_graph, bu_graph, texts_new)
            texts_new.append("")
            return texts_new, txt, m_idx, new_td.add_self_loop(), new_bu.add_self_loop(), prob, aug_type
        r = random.random()
        if r < 0.33:
            aug_type, txt, prob, m_idx, new_td, new_bu = self.add_node(td_graph, bu_graph, texts_new)
            texts_new.append("")
            return texts_new, txt, m_idx, new_td.add_self_loop(), new_bu.add_self_loop(), prob, aug_type
        elif r < 0.66:
            aug_type, txt, prob, m_idx = self.modify_node(td_graph, texts_new)
            return texts_new, txt, m_idx, td_graph.add_self_loop(), bu_graph.add_self_loop(), prob, aug_type
        else:
            aug_type, txt, prob, m_idx, new_td, new_bu = self.delete_node(td_graph)
            texts_new.pop(m_idx)
            return texts_new, txt, -1, new_td.add_self_loop(), new_bu.add_self_loop(), prob, aug_type

    def leaf_idxs(self, tree: dgl.DGLGraph):
        s, d = tree.remove_self_loop().edges()
        source = s.cpu().tolist()
        leaf_ids = [idx for idx in range(tree.number_of_nodes()) if idx not in source]
        return leaf_ids

    def add_node(self, td_tree: dgl.DGLGraph, bu_tree: dgl.DGLGraph, texts):
        td, bu = copy.deepcopy(td_tree), copy.deepcopy(bu_tree)
        node_cnt = td.number_of_nodes()
        leaf_ids = self.leaf_idxs(td_tree)
        idx = random.sample(range(node_cnt), 1)[0]
        td.add_edges(idx, node_cnt)
        bu.add_edges(node_cnt, idx)
        if idx in leaf_ids:
            accep_term_2 = (node_cnt * self.beam_K) * 1.0 / len(leaf_ids)
        else:
            accep_term_2 = (node_cnt * self.beam_K) * 1.0 / (len(leaf_ids) + 1)
        prompt_text = " ".join([texts[0], texts[idx]])
        prompt_text = re.sub(r"[@#] ", '', prompt_text)
        return 0, prompt_text, accep_term_2, node_cnt, td, bu

    def modify_node(self, td_tree: dgl.DGLGraph, texts, node_idx=None):
        s, d = td_tree.remove_self_loop().edges()
        idx = random.sample(range(len(d)), 1)[0] if node_idx is None else node_idx
        accep_term_2 = 1.0
        prompt_text = " ".join([texts[0], texts[s[idx].data.item()]])
        prompt_text = re.sub(r"[@#] ", '', prompt_text)
        return 1, prompt_text, accep_term_2, idx

    def delete_node(self, td_tree: dgl.DGLGraph, node_idx=None):
        leaf_ids = self.leaf_idxs(td_tree)
        node_cnt = td_tree.number_of_nodes()
        idx = random.sample(leaf_ids, 1)[0] if node_idx is None else node_idx
        s, d = td_tree.remove_self_loop().edges()
        edges = [(start if start < idx else start - 1, end if end < idx else end - 1)
                 for start, end in zip(s.cpu().tolist(), d.cpu().tolist()) if end != idx]
        src = np.array([item[0] for item in edges])
        dst = np.array([item[1] for item in edges])
        td_tree_new = dgl.graph((src, dst), num_nodes=node_cnt - 1)
        bu_tree_new = dgl.graph((dst, src), num_nodes=node_cnt - 1)
        if node_cnt > 2:
            accep_term_2 = (len(leaf_ids) * 1.0) / ((node_cnt - 1) * self.beam_K)
        else:
            accep_term_2 = (len(leaf_ids) * 3.0) / ((node_cnt - 1) * self.beam_K)
            # when the graph only contains one root node, there only one jump-back operation, i.e., add node, as we
            # cannot delete the root node and cannot modify the content in the root node.
        return 2, None, accep_term_2, idx, td_tree_new.add_self_loop(), bu_tree_new.add_self_loop()


def get_logpath(suffix=""):
    """Create a logpath and return it.

    Args:
        suffix (str, optional): suffix to add to the output. Defaults to "".

    Returns:
        str: Path to the logfile (output of Cockpit).
    """
    save_dir = os.path.join(os.getcwd(), "logfiles")
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, f"cockpit_output{suffix}")
    return log_path

if __name__ == '__main__':
    data_dir1 = r"../MetaGenerator/pheme-rnr-dataset/"
    # os.environ['CUDA_VISIBLE_DEVICES'] = "0" 

    events_list = ['charliehebdo', 'ferguson', 'germanwings-crash', 'ottawashooting', 'sydneysiege']
    # for domain_ID in range(5):
    domain_ID = 0
    source_events = [os.path.join(data_dir1, dname)
                     for idx, dname in enumerate(events_list) if idx != domain_ID]
    # source_events = [os.path.join(data_dir1, dname)
    #                  for idx, dname in enumerate(events_list)]
    target_events = [os.path.join(data_dir1, events_list[domain_ID])]
    test_event_name = events_list[domain_ID]

    tr, _, dev, meta_dev, te = load_data(
        source_events, target_events, 0, unlabeled_ratio=-1
    )

    # events_list_all = [os.path.join(data_dir1, dname)
    #                    for idx, dname in enumerate(events_list)]
    # tr, _, dev, meta_dev, te = load_data_all(
    #     events_list_all, 0, unlabeled_ratio=-1
    # )
    #
    # root_dir = "indomain"
    # test_event_name = "pheme"

    log_dir = f"./{test_event_name}/"

    print("%s : (dev event)/(test event)/(train event) = %3d/%3d/%3d" % (
        test_event_name, len(dev), len(te), len(tr)))
    # print("\n\n===========%s Train===========\n\n" % te.data[te.data_ID[0]]['event'])
    print("\n\n===========%s Out Domain Train===========\n\n" % test_event_name)

    bertPath = r"../MetaGenerator/models/bert-base-uncased"
    model = obtain_Transformer(bertPath)
    tr.initGraph()
    dev_eval = TransformerEvaluator(dev, batch_size=20)
    te_eval = TransformerEvaluator(te, batch_size=20)
    trainer = MCMCBiGCNTrainer(log_dir, None, model_rename=True, beam_K=3, batch_dir=f"./aug_dir4")
    # trainer.fit(model, tr, dev_eval, te_eval, batch_size=32, grad_accum_cnt=1, learning_rate=2e-5,
    #             max_epochs=20,
    #             model_file=os.path.join(log_dir, f'BiGCN_{test_event_name}.pkl'))
    # trainer.dataset2dataloader2(model, tr, shuffle=True, batch_size=32)
    # trainer.meta_learning(model, train_set=tr, meta_dev=meta_dev, dev_eval=dev_eval, test_eval=te_eval, batch_size=32,
    #                       grad_accum_cnt=1, valid_every=100,
    #                       max_epochs=20, learning_rate=2e-5, model_lr=2e-5,
    #                       model_file=os.path.join(log_dir, f'BiGCN_{test_event_name}.pkl'), weight_lr=0.1, meta_step=10)
    
    trainer.validate_cpt(model,test_eval=te_eval)
