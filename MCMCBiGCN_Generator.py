import copy
import re
import sys, nltk, dgl, torch, random, os, pickle
import time
from typing import List

import numpy as np
from torch_geometric.data import Data
from transformers import TextGenerationPipeline, pipeline, GPT2Tokenizer, GPT2LMHeadModel

from Data.AmazonLoader import transIrregularWord
from Data.BiGCN_Data_Utils.twitterloader import BiGCNTwitterSet
from Data.BiGCN_Dataloader import FastBiGCNDataset, load_data
from HierarchicalTransformer import obtain_Transformer
from TrainingEnv import VirtualModel


class MetaMCMCDataset(FastBiGCNDataset):

    def __init__(self, dataset: BiGCNTwitterSet, batch_size=20, beam_K=3, batch_dir="./aug_dir"):
        super(MetaMCMCDataset, self).__init__(batch_size=batch_size)
        self.instance_weights = torch.ones([len(self.data_len)], dtype=torch.float32, device=torch.device('cuda:0'))
        self.batch_dir = batch_dir
        self.beam_K = beam_K
        self.dataset = dataset
        self.inherit_data()

    def inherit_data(self):
        data_size = len(self.dataset)
        new_idxs = list(random.sample(list(range(data_size)), data_size))
        self.data_ID = [self.dataset.data_ID[idx] for idx in new_idxs]
        self.data_len = [self.dataset.data_len[idx] for idx in new_idxs]
        self.data_y = [self.dataset.data_y[idx] for idx in new_idxs]
        self.data = {ID: self.dataset.data[ID] for ID in self.data_ID}
        self.instance_weights = torch.ones([len(self.data_ID)], dtype=torch.float32, device=torch.device('cuda:0'))
        self.initGraph()

    def set_instance_weights(self, weights_list, idxs_list):
        self.instance_weights[torch.cat(idxs_list)] = torch.squeeze(torch.cat(weights_list))

    def init_instance_weights(self):
        self.instance_weights = torch.ones([len(self.data_ID)], dtype=torch.float32, device=torch.device('cuda:0'))

    def get_instance_weights(self):
        return self.instance_weights

    def load_data_fast(self, data_prefix="../data/train", min_len=-1):

        super().load_data_fast(data_prefix)
        self.instance_weights = torch.ones([len(self.data_ID)], dtype=torch.float32, device=torch.device('cuda:0'))

    def __getitem__(self, index):
        d_ID = self.data_ID[index]
        if not hasattr(self, "g_TD"):
            self.g_TD = {}
        if not hasattr(self, "g_BU"):
            self.g_BU = {}
        if not hasattr(self, "lemma_text"):
            self.lemma_text = {}

        if d_ID in self.g_TD and d_ID in self.g_BU:
            g_TD, g_BU = self.g_TD[d_ID], self.g_BU[d_ID]
        else:
            g_TD, g_BU = self.construct_graph(index, d_ID)
            self.g_TD[d_ID] = g_TD
            self.g_BU[d_ID] = g_BU
        sents = [" ".join(sent) for sent in self.data[d_ID]['text']]
        weight = self.instance_weights[index]
        return Data(g_TD=dgl.add_self_loop(g_TD), g_BU=dgl.add_self_loop(g_BU),
                    data_y=self.data_y[index], topic_label=self.data[self.data_ID[index]]['topic_label'], text=sents,
                    data_len=self.data_len[index], weight=weight, index=index)

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

    def MCMC_Augmentation(self, model: VirtualModel, lm_generator: TextGenerationPipeline,
                          max_step=5, early_stop_confidence=0.5, batch_idxs: List = None,
                          batch_size=10, temperature=1.0, kappa=0.5, tr_data_len=4600):
        if not hasattr(self, 'init_length'):
            self.init_length = len(self.data_ID)
        if batch_idxs is None:
            batch_idxs = random.sample(range(tr_data_len), batch_size)
        else:
            assert len(batch_idxs) == batch_size
        d_ID_list = [self.data_ID[idx] for idx in batch_idxs]
        # new_data_list = [copy.deepcopy(self.data[d_ID]) for d_ID in d_ID_list]
        try:
            new_data_list = [copy.deepcopy(self.data[d_ID]) for d_ID in d_ID_list]
        except:
            import pdb
            pdb.set_trace()
            raise

        timestamp = int(time.time())
        batch_IDs = [str(timestamp + kk) for kk in range(len(batch_idxs))]
        print("batch_IDs : ", batch_IDs)
        self.data_ID.extend(batch_IDs)
        for kk, bID in enumerate(batch_IDs):
            self.data[bID] = new_data_list[kk]
            self.g_TD[bID] = copy.deepcopy(self.g_TD[self.data_ID[batch_idxs[kk]]])
            self.g_BU[bID] = copy.deepcopy(self.g_BU[self.data_ID[batch_idxs[kk]]])
        self.data_len.extend(
            [self.data_len[idx] for idx in batch_idxs]
        )
        self.data_y.extend(
            [self.data_y[idx] for idx in batch_idxs]
        )
        batch_labels = torch.tensor(
            [self.data_y[idx] for idx in batch_idxs],
            device=self.device,
            dtype=torch.float32
        )
        texts_cur = [[" ".join(self.lemma(self.data[b_ID]['text'][j])) for j in range(self.data_len[index])]
                     for b_ID, index in zip(batch_IDs, batch_idxs)]
        TD_Graphs_cur = [copy.deepcopy(self.g_TD[self.data_ID[b_idx]]).add_self_loop() for b_idx in batch_idxs]
        BU_Graphs_cur = [copy.deepcopy(self.g_BU[self.data_ID[b_idx]]).add_self_loop() for b_idx in batch_idxs]
        confidence_cur = self.model_logits(model, texts_cur, TD_Graphs_cur, BU_Graphs_cur, batch_labels)
        best_confidence = confidence_cur.clone()
        for step in range(max_step):
            if confidence_cur.max() < early_stop_confidence:
                return
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
            confidence_next = self.model_logits(model, texts_next, TD_Graphs_next,
                                                BU_Graphs_next, batch_labels)
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
            accept_idxs = torch.arange(batch_size)[
                (accept_ratio.cpu() - torch.rand(batch_size)).__gt__(0.0)
            ].tolist()

            for a_idx in accept_idxs:
                TD_Graphs_cur[a_idx] = TD_Graphs_next[a_idx]
                BU_Graphs_cur[a_idx] = BU_Graphs_next[a_idx]
                texts_cur[a_idx] = texts_next[a_idx]

            preserve_idx = torch.arange(batch_size)[
                (confidence_next - best_confidence).__gt__(0).cpu()
            ].tolist()
            for p_idx in preserve_idx:
                self.data[batch_IDs[p_idx]]['text'] = [[item for item in re.split(r'[ ]+', s) if item.strip(" ") != '']
                                                       for s in texts_next[p_idx]]
                self.data_len[len(self.data_len) - batch_size + p_idx] = len(texts_next[p_idx])
                self.g_BU[batch_IDs[p_idx]] = BU_Graphs_next[p_idx]
                self.g_TD[batch_IDs[p_idx]] = TD_Graphs_next[p_idx]
            best_confidence = torch.where(confidence_next > best_confidence, confidence_next, best_confidence)
            confidence_cur = confidence_next
        self.extract_augmented_batch(batch_size)

    def extract_augmented_batch(self, batch_size):
        items = [self.__getitem__(index) for index in range(-1 * batch_size, 0, 1)]
        batch = self.collate_fn(items)
        t = int(time.time())
        for _ in range(batch_size):
            d_ID = self.data_ID.pop(-1)
            self.data_len.pop(-1)
            self.data_y.pop(-1)
            self.g_TD.pop(d_ID)
            self.g_BU.pop(d_ID)
            self.data.pop(d_ID)
        with open(os.path.join(self.batch_dir, f"{t}.batch"), "wb") as fw:
            pickle.dump(batch, fw, protocol=pickle.HIGHEST_PROTOCOL)
        signal_file = os.path.join(self.batch_dir, f"{t}.ok")
        os.system(f"touch {signal_file}")

    def model_logits(self, model: VirtualModel, texts: List, TD_graphs: List, BU_graphs: List,
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
        return energy

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

    def leaf_idxs(self, tree:dgl.DGLGraph):
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


def get_new_model_path(model_dir):
    print("model_dir", model_dir)
    signal_files = [os.path.join(model_dir, fname) for fname in os.listdir(model_dir) if fname[-3:] == '.ok']
    if len(signal_files) == 0:
        return None
    signal_files.sort()
    latest_signal = signal_files.pop(-1)
    backup_dir = latest_signal.replace(".ok", '')
    model_name = [fname for fname in os.listdir(backup_dir) if '.modelname' in fname][-1]
    # os.system(f"rm -f {os.path.join(backup_dir, model_name)}")
    model_path = os.path.join(backup_dir, model_name.replace(".modelname", ".pkl"))
    if len(signal_files) != 0:
        for signal_file in signal_files:
            os.system(f"rm -rf {signal_file.replace('.ok', '')}")
    return model_path


if __name__ == '__main__':
    # log_dir = str(__file__).rstrip(".py")
    running_dir = os.getcwd() + "/tmp/"
    data_dir = r"D:\科研\MetaGenerator\pheme-rnr-dataset"
    events_list = ['charliehebdo', 'ferguson', 'germanwings-crash', 'ottawashooting', 'sydneysiege']
    # for domain_ID in range(5):
    domain_ID = 0
    source_events = [os.path.join(data_dir, dname)
                     for idx, dname in enumerate(events_list) if idx != domain_ID]
    target_events = [os.path.join(data_dir, events_list[domain_ID])]
    test_event_name = events_list[domain_ID]
    tr, _, dev, meta_dev, te = load_data(
        source_events, target_events, 0, unlabeled_ratio=-1
    )
    log_dir = f"./{test_event_name}/"
    print("%s : (dev event)/(test event)/(train event) = %3d/%3d/%3d" % (
        test_event_name, len(dev), len(te), len(tr)))
    print("\n\n===========%s Train===========\n\n" % te.data[te.data_ID[0]]['event'])

    tr_data_len = len(tr.data)
    print("tr_data_len", tr_data_len)
    tr = MetaMCMCDataset(tr, batch_size=32, beam_K=3, batch_dir=f"./aug_dir")
    tr_data_len = len(tr.data)
    print("tr_data_len", tr_data_len)

    bertPath = r"D:\科研\MCMC\models\bert-base-uncased"
    model = obtain_Transformer(bertPath)
    model.load_model(os.path.join(log_dir, f'BiGCN_{test_event_name}.pkl'))
    tok = GPT2Tokenizer.from_pretrained("D:\科研\MCMC\gpt2")
    # tok.model_max_length = 128
    gpt2 = GPT2LMHeadModel.from_pretrained("D:\科研\MCMC\gpt2").cuda()
    from transformers import pipeline

    lm_generator = pipeline('text-generation', model=gpt2, tokenizer=tok, device=0 if torch.cuda.is_available() else -1)

    for i in range(1500):
        tr.MCMC_Augmentation(model, lm_generator, max_step=5, batch_size=32, temperature=0.1, tr_data_len=tr_data_len)
        new_path = get_new_model_path(running_dir)
        if new_path is not None:
            model.load_model(new_path)
        torch.cuda.empty_cache()

        cached_batches = [fname for fname in os.listdir("./aug_dir") if ".batch" in fname]
        if len(cached_batches) > 50:
            os.system("rm -rf ./aug_dir")
            time.sleep(1200)
            sys.exit(0)
