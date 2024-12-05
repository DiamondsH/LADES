import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import os
import fitlog
import sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from SentModel.sent2vec_utils import grad_reverse
import dgl
import torch.nn.functional as F
import numpy as np

class SequenceClassificationTrainer(nn.Module):
    def __init__(self, sent2vec, propagation, classifier=None, label_cnt=2, batch_size=5, grad_accum_cnt=4):
        super(SequenceClassificationTrainer, self).__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.sent2vec = sent2vec.to(self.device)
        self.prop_model = propagation.to(self.device)
        if classifier is not None:
            self.classifier = classifier.to(self.device)
        else:
            self.classifier = nn.Linear(self.prop_model.prop_hidden_size, label_cnt)
        self.nll_loss_fn = nn.NLLLoss()
        self.batch_size = batch_size
        self.grad_accum_cnt = grad_accum_cnt

    def seq2sents(self, seqs):
        sent_list = [sent for seq in seqs for sent in seq]
        seq_len = [len(seq) for seq in seqs]
        return sent_list, seq_len

    def forward(self, seqs):
        sents, seq_len = self.seq2sents(seqs)
        sent_vecs = self.sent2vec(sents)
        seq_tensors = [sent_vecs[sum(seq_len[:idx]):sum(seq_len[:idx]) + seq_len[idx]] for idx, s_len in
                       enumerate(seq_len)]
        seq_outs = self.prop_model(seq_tensors)
        preds = self.classifier(seq_outs).softmax(dim=1)
        return preds

    def Loss(self, seqs, labels):
        preds = self.forward(seqs)
        loss = self.nll_loss_fn(preds.log(), labels)
        acc = accuracy_score(labels.cpu().numpy(), preds.argmax(dim=1).cpu().numpy())
        return loss, acc

    def train_iters(self, train_loader, dev_loader, te_loader,
                    valid_every=100, max_epochs=10, lr_discount=1.0,
                    best_valid_acc=0.0, best_test_acc=0.0, best_valid_test_acc=0.0,
                    log_dir="../logs/", log_suffix="_PropTrainer", model_file=""):
        optim = torch.optim.Adam([
            {'params': self.sent2vec.parameters(), 'lr': 5e-5 * lr_discount / self.grad_accum_cnt},
            {'params': self.prop_model.parameters(), 'lr': 1e-3 * lr_discount / self.grad_accum_cnt},
            {'params': self.classifier.parameters(), 'lr': 1e-3 * lr_discount / self.grad_accum_cnt}
        ]
        )
        
        counter = 0
        optim.zero_grad()
        self.train()
        sum_loss, sum_acc = 0.0, 0.0
        for epoch in range(max_epochs):
            for step, batch in enumerate(train_loader):
                loss, acc = self.Loss(batch[0], batch[1].to(self.device))
                loss.backward()
                torch.cuda.empty_cache()
                if step % self.grad_accum_cnt == 0:
                    optim.step()
                    optim.zero_grad()
                sum_loss += loss
                sum_acc += acc
                if (step + 1) % self.grad_accum_cnt == 0:
                    print('%6d | %6d  [%3d | %3d], loss/acc = %6.8f/%6.7f, best_valid_acc:%6.7f ' % (
                        step, len(train_loader),
                        epoch, max_epochs,
                        sum_loss / self.grad_accum_cnt, sum_acc / self.grad_accum_cnt,
                        best_valid_acc
                    )
                          )
                    fitlog.add_metric(
                        {"train": {"acc": sum_acc / self.grad_accum_cnt, "loss": sum_loss / self.grad_accum_cnt}},
                        step=counter
                    )
                    sum_loss, sum_acc = 0.0, 0.0
                    counter += 1
                if (step + 1) % (valid_every * self.grad_accum_cnt) == 0:
                    val_acc, val_loss = self.valid(dev_loader)
                    test_acc, test_loss = self.valid(te_loader)
                    self.train()
                    best_test_acc = test_acc if best_test_acc < test_acc else best_test_acc
                    print(
                        '##### %6d | %6d, [%3d | %3d], val_loss|val_acc = %6.8f/%6.7f, te_loss|te_acc = %6.8f/%6.7f, best_valid_acc/related_test_acc= %6.7f/%6.7f, best_test_acc=%6.7f' % (
                            step, len(train_loader),
                            epoch, max_epochs,
                            val_loss, val_acc,
                            test_loss, test_acc,
                            best_valid_acc, best_valid_test_acc,
                            best_test_acc
                        )
                    )
                    fitlog.add_metric(
                        {"valid":{"acc":val_acc, "loss":val_loss}}, 
                        step=counter
                    )
                    if val_acc > best_valid_acc:
                        best_valid_acc = val_acc
                        best_valid_test_acc = test_acc
                        self.save_model(model_file)
        return best_valid_acc, best_test_acc, best_valid_test_acc

    def valid(self, data_loader:DataLoader, pretrained_file=None, all_metrics=False):
        self.eval()
        if pretrained_file is not None and os.path.exists(pretrained_file):
            self.load_model(pretrained_file)
        labels = []
        preds = []
        with torch.no_grad():
            for batch in data_loader:
                pred = self.forward(batch[0])
                torch.cuda.empty_cache()
                preds.append(pred)
                labels.append(batch[1])
            pred_tensor = torch.cat(preds, dim=0)
            label_tensor = torch.cat(labels, dim=0)
            val_acc = accuracy_score(label_tensor.numpy(),
                                     pred_tensor.cpu().argmax(dim=1).numpy())
            val_loss = self.nll_loss_fn(pred_tensor.to(self.device).log(), label_tensor.to(self.device))
        self.train()
        if all_metrics:
            val_prec = precision_score(label_tensor.numpy(),
                                       pred_tensor.cpu().argmax(dim=1).numpy())
            val_recall = recall_score(label_tensor.numpy(),
                                       pred_tensor.cpu().argmax(dim=1).numpy())
            val_f1 = f1_score(label_tensor.numpy(),
                               pred_tensor.cpu().argmax(dim=1).numpy())
            return val_acc, val_loss, val_prec, val_recall, val_f1
        else:
            return val_acc, val_loss

    def save_model(self, model_file):
        torch.save(
            {
                "sent2vec": self.sent2vec.state_dict(),
                "prop_model": self.prop_model.state_dict(),
                "classifier": self.classifier.state_dict()
            },
            model_file
        )

    def load_model(self, model_file):
        if os.path.exists(model_file):
            checkpoint = torch.load(model_file)
            
            self.classifier.load_state_dict(checkpoint["classifier"])
            self.prop_model.load_state_dict(checkpoint['prop_model'])
        else:
            print("Error: pretrained file %s is not existed!" % model_file)
            sys.exit()

class AdverSequenceClassificationTrainer(SequenceClassificationTrainer):
    def __init__(self, sent2vec, propagation, classifier=None, label_cnt=2, batch_size=5, grad_accum_cnt=4):
        super(AdverSequenceClassificationTrainer, self).__init__(sent2vec, propagation, classifier, label_cnt, batch_size, grad_accum_cnt)

    def forward(self, seqs):
        sents, seq_len = self.seq2sents(seqs)
        sent_vecs = self.sent2vec(sents)
        seq_tensors = [sent_vecs[sum(seq_len[:idx]):sum(seq_len[:idx]) + seq_len[idx]] for idx, s_len in
                       enumerate(seq_len)]
        seq_outs = self.prop_model(seq_tensors)
        seq_outs = grad_reverse(seq_outs)
        preds = self.classifier(seq_outs).softmax(dim=1)
        return preds


class GraphAutoEncoderTrainer(nn.Module):
    def __init__(self, sent2vec, graph_model, batch_size=1, grad_accum_cnt=1):
        super(GraphAutoEncoderTrainer, self).__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.sent2vec = sent2vec
        self.graph_model = graph_model
        self.loss_fn = nn.BCELoss(reduce=False)
        self.batch_size = batch_size
        self.grad_accum_cnt = grad_accum_cnt

    def graphs2BigGraph(self, graphs):
        sents = [sent for graph in graphs for sent in graph.ndata.pop('sentence')]
        bg = dgl.batch(graphs)
        bg.ndata['sentence'] = np.array(sents)
        return bg

    def forward(self, graph):
        sent_vecs = self.sent2vec(graph.ndata.pop('sentence'))
        H_Vec = self.graph_model(graph, sent_vecs)
        dot = torch.matmul(H_Vec, H_Vec.transpose(0, 1))
        preds = F.sigmoid(dot)
        return preds

    def Loss(self, graph, edges):
        preds = self.forward(graph)
        loss_mtx = self.loss_fn(preds, edges.to(self.device))
        loss_weights = edges.to(self.device) * 9 + 1
        loss = (loss_mtx * loss_weights).mean()
        acc = accuracy_score(preds.gt(0.5).int().reshape(-1).cpu().numpy(), edges.reshape(-1).cpu().numpy())
        return loss, acc

    def train_iters(self, train_set, dev_set, test_set,
                    valid_every=100, max_epochs=10, lr_discount=1.0,
                    log_dir="../logs/", log_suffix="_RumorDetection", model_file=""):
        fitlog.set_log_dir("%s/" % log_dir, new_log=True)
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True,
                                  collate_fn=train_set.collate_raw_batch)
        optim = torch.optim.Adam([
            {'params': self.sent2vec.parameters(), 'lr': 2e-5 * lr_discount / self.grad_accum_cnt},
            {'params': self.graph_model.parameters(), 'lr': 2e-5 * lr_discount / self.grad_accum_cnt}
        ]
        )

        best_valid_acc, best_test_acc, best_valid_test_acc = 0.0, 0.0, 0.0
        counter = 0
        optim.zero_grad()
        self.train()
        sum_loss, sum_acc = 0.0, 0.0
        for epoch in range(max_epochs):
            for step, batch in enumerate(train_loader):
                batch_graph = self.graphs2BigGraph(batch[0])
                adj_mtx = batch_graph.adjacency_matrix().to_dense()
                print("adj_mtx shape:", adj_mtx.shape)
                loss, acc = self.Loss(batch_graph, adj_mtx)
                loss.backward()
                torch.cuda.empty_cache()
                if step % self.grad_accum_cnt == 0:
                    optim.step()
                    optim.zero_grad()
                sum_loss += loss
                sum_acc += acc
                if (step + 1) % self.grad_accum_cnt == 0:
                    print('%6d | %6d  [%3d | %3d], loss/acc = %6.8f/%6.7f, best_valid_acc:%6.7f ' % (
                        step, len(train_loader),
                        epoch, max_epochs,
                        sum_loss / self.grad_accum_cnt, sum_acc / self.grad_accum_cnt,
                        best_valid_acc
                    )
                          )
                    fitlog.add_metric(
                        {"train": {"acc": sum_acc / self.grad_accum_cnt, "loss": sum_loss / self.grad_accum_cnt}},
                        step=counter
                    )
                    sum_loss, sum_acc = 0.0, 0.0
                    counter += 1
                if (step + 1) % (valid_every * self.grad_accum_cnt) == 0:
                    val_acc, val_loss = self.valid(dev_set)
                    test_acc, test_loss = self.valid(test_set)
                    self.train()
                    best_test_acc = test_acc if best_test_acc < test_acc else best_test_acc
                    print(
                        '##### %6d | %6d, [%3d | %3d], val_loss|val_acc = %6.8f/%6.7f, te_loss|te_acc = %6.8f/%6.7f, best_valid_acc/related_test_acc= %6.7f/%6.7f, best_test_acc=%6.7f' % (
                            step, len(train_loader),
                            epoch, max_epochs,
                            val_loss, val_acc,
                            test_loss, test_acc,
                            best_valid_acc, best_valid_test_acc,
                            best_test_acc
                        )
                    )
                    fitlog.add_metric(
                        {"valid": {"acc": val_acc, "loss": val_loss}},
                        step=counter
                    )
                    if val_acc > best_valid_acc:
                        best_valid_acc = val_acc
                        best_valid_test_acc = test_acc
                        self.save_model(model_file)
        fitlog.add_best_metric({"%s" % log_suffix: {"best_valid_acc": best_valid_test_acc,
                                                    "best_valid_test_acc": best_valid_test_acc,
                                                    "best_test_acc": best_test_acc}})
        fitlog.finish()

    def valid(self, data_set, pretrained_file=None, all_metrics=False):
        self.eval()
        if pretrained_file is not None and os.path.exists(pretrained_file):
            self.load_model(pretrained_file)
        labels = []
        preds = []
        sum_loss = 0.
        with torch.no_grad():
            for i in range(len(data_set)):
                batch = data_set.__getitem__(i)
                adj_mtx = batch[0].adjacency_matrix().to_dense()
                pred = self.forward(batch[0])
                loss_mtx = self.loss_fn(pred, adj_mtx.to(self.device))
                loss_weights = adj_mtx.to(self.device) * 9 + 1
                loss = (loss_mtx * loss_weights).mean()

                torch.cuda.empty_cache()
                sum_loss += loss
                preds.append(pred.gt(0.5).int().reshape(-1))
                labels.append(adj_mtx.reshape(-1))

            pred_tensor = torch.cat(preds, dim=0)
            label_tensor = torch.cat(labels, dim=0)
            val_loss = sum_loss * 1.0 / (i + 1)
            val_acc = accuracy_score(label_tensor.numpy(),
                                     pred_tensor.cpu().numpy())
        if all_metrics:
            val_prec = precision_score(label_tensor.numpy(),
                                       pred_tensor.cpu().numpy())
            val_recall = recall_score(label_tensor.numpy(),
                                      pred_tensor.cpu().numpy())
            val_f1 = f1_score(label_tensor.numpy(),
                              pred_tensor.cpu().numpy())
            return val_acc, val_loss, val_prec, val_recall, val_f1
        else:
            return val_acc, val_loss

    def save_model(self, model_file):
        sent_model_file = "%s_sent.pkl" % (model_file.rstrip(".pkl"))
        self.sent2vec.save_model(sent_model_file)
        torch.save(
            {

                "graph_model": self.graph_model.state_dict()
            },
            model_file
        )

    def load_model(self, model_file):
        if os.path.exists(model_file):
            sent_model_file = "%s_sent.pkl" % (model_file.rstrip(".pkl"))
            self.sent2vec.load_model(sent_model_file)
            checkpoint = torch.load(model_file)

            self.graph_model.load_state_dict(checkpoint['graph_model'])
        else:
            print("Error: pretrained file %s is not existed!" % model_file)
            sys.exit()