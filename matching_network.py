import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_

from collections import OrderedDict
from torchmeta.utils import gradient_update_parameters
from meta_utils import tensors_to_device, compute_accuracy

from pdb import set_trace

from GPUtil import showUtilization as gpu_usage
__all__ = ['ModelAgnosticMetaLearning', 'MAML', 'FOMAML']


class MatchingNetTrainer(object):

    def __init__(self,args, model, optimizer=None, step_size=0.1, 
                 num_adaptation_steps=1,
                 loss_fn=F.cross_entropy, fce = True):

        self.args = args

        self.model = model.cuda().float()
        self.optimizer = optimizer
        self.step_size = step_size
        self.loss_fn = loss_fn
        self.device = 'cuda'
        self.fce = fce

    def train(self, meta_train_dataloader, max_batches= 500, verbose = True, **kwargs):

        with tqdm(total=max_batches, disable=False, **kwargs) as pbar:
            for results in self.train_iter(meta_train_dataloader, max_batches = max_batches):
                pbar.update(1)
                postfix = {'loss': '{0:.4f}'.format(results['loss'])}

                if 'accuracies' in results:
                    postfix['accuracy'] = results['accuracies']

                pbar.set_postfix(**postfix)

    def train_iter(self, dataloader,max_batches = 500):
        num_batches = 0
        self.model.train()
        while num_batches < max_batches:
            for batch in dataloader:
                #gpu_usage()
                if num_batches >= max_batches:
                    break

                #if self.scheduler is not None:
                #    self.scheduler.step(epoch=num_batches)

                self.optimizer.zero_grad()
                batch = tensors_to_device(batch, device=self.device)
                
                #self.model.double()
                loss, results = self.get_loss(batch)

                yield results
                loss.backward()
                #self.model.float()
                clip_grad_norm_(self.model.parameters(),1)
                self.optimizer.step()
                num_batches +=1

    def evaluate(self, dataloader, max_batches, verbose = True, **kwargs):
        mean_outer_loss, mean_accuracy, count = 0., 0., 0
        with tqdm(total=max_batches, disable= False, **kwargs) as pbar:
            for results in self.evaluate_iter(dataloader, max_batches=max_batches):
                pbar.update(1)
                count += 1
                mean_outer_loss += (results['loss']
                    - mean_outer_loss) / count
                postfix = {'loss': '{0:.4f}'.format(mean_outer_loss)}
                if 'accuracies' in results:
                    mean_accuracy += (np.mean(results['accuracies'])
                        - mean_accuracy) / count
                    postfix['accuracy'] = '{0:.4f}'.format(mean_accuracy)
                pbar.set_postfix(**postfix)

        mean_results = {'mean_outer_loss': mean_outer_loss}
        if 'accuracies' in results:
            mean_results['accuracies'] = mean_accuracy

        return mean_results

    def evaluate_iter(self, dataloader, max_batches=500):
        num_batches = 0
        self.model.eval()
        while num_batches < max_batches:
            for batch in dataloader:
                #gpu_usage()
                if num_batches >= max_batches:
                    break

                batch = tensors_to_device(batch, device=self.device)
                #self.model.double()
                _, results = self.get_loss(batch)
                #self.model.float()
                yield results

                num_batches += 1

        
    def get_loss(self,batch):
        EPSILON = 1e-8
        _, test_targets = batch['test']
        num_tasks = test_targets.size(0)
        
        mean_loss = torch.tensor(0., device=self.device)
        
        results = {
            'loss': np.zeros((1,), dtype=np.float32),
             }
        train_inputs, train_targets = batch['train']
        test_inputs, test_targets = batch['test']

        support = self.model.encoder(train_inputs.squeeze(0))
        queries   = self.model.encoder(test_inputs.squeeze(0))
        if self.fce:
            support, _, _ = self.model.g(support.unsqueeze(1))
            support = support.squeeze(1)
            queries = self.model.f(support.float(), queries.float())

        distances = self.pairwise_distances(queries, support, 'l2')
        attention = (-distances).softmax(dim=1)
            
        y_pred = self.matching_net_predictions(attention, train_targets.squeeze(0) )    
            
        clipped_y_pred = y_pred.clamp(EPSILON, 1 - EPSILON)
        loss = self.loss_fn(clipped_y_pred.log(), test_targets.squeeze(0))       


        results['accuracies'] = compute_accuracy(clipped_y_pred, test_targets)
        
        results['loss'] = loss.item()

        return loss, results

    def pairwise_distances(self,x: torch.Tensor,
                       y: torch.Tensor,
                       matching_fn: str) -> torch.Tensor:
        """Efficiently calculate pairwise distances (or other similarity scores) between
        two sets of samples.

        # Arguments
            x: Query samples. A tensor of shape (n_x, d) where d is the embedding dimension
            y: Class prototypes. A tensor of shape (n_y, d) where d is the embedding dimension
            matching_fn: Distance metric/similarity score to compute between samples
        """
        n_x = x.shape[0]
        n_y = y.shape[0]
        EPSILON = 1e-8
        if matching_fn == 'l2':
            distances = (
                x.unsqueeze(1).expand(n_x, n_y, -1) -
                y.unsqueeze(0).expand(n_x, n_y, -1)
            ).pow(2).sum(dim=2)
            return distances
        elif matching_fn == 'cosine':
            normalised_x = x / (x.pow(2).sum(dim=1, keepdim=True).sqrt() + EPSILON)
            normalised_y = y / (y.pow(2).sum(dim=1, keepdim=True).sqrt() + EPSILON)

            expanded_x = normalised_x.unsqueeze(1).expand(n_x, n_y, -1)
            expanded_y = normalised_y.unsqueeze(0).expand(n_x, n_y, -1)

            cosine_similarities = (expanded_x * expanded_y).sum(dim=2)
            return 1 - cosine_similarities

        elif matching_fn == 'dot':
            expanded_x = x.unsqueeze(1).expand(n_x, n_y, -1)
            expanded_y = y.unsqueeze(0).expand(n_x, n_y, -1)

            return -(expanded_x * expanded_y).sum(dim=2)
        else:
            raise(ValueError('Unsupported similarity function'))

    def matching_net_predictions(self,attention: torch.Tensor, support_labels) -> torch.Tensor:
        """Calculates Matching Network predictions based on equation (1) of the paper.

        The predictions are the weighted sum of the labels of the support set where the
        weights are the "attentions" (i.e. softmax over query-support distances) pointing
        from the query set samples to the support set samples.

        # Arguments
            attention: torch.Tensor containing softmax over query-support distances.
            Should be of shape (q * k, k * n)
            n: Number of support set samples per class, n-shot
            k: Number of classes in the episode, k-way
            q: Number of query samples per-class

        # Returns
            y_pred: Predicted class probabilities
        """
        k = self.args.num_ways
        n = self.args.num_shots
        q = self.args.num_shots_test

        if attention.shape != (q * k, k * n):
            raise(ValueError(f'Expecting attention Tensor to have shape (q * k, k * n) = ({q * k, k * n})'))

        y_onehot = F.one_hot(support_labels, 5).float()
        y_pred = torch.mm(attention, y_onehot.cuda())
        return y_pred

